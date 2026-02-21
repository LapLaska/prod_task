from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import faiss
from .utils import get_torch_device


@dataclass
class OfferTextArtifacts:
    offer_ids: np.ndarray
    embeddings: np.ndarray
    faiss_index: faiss.IndexFlatIP


class OfferTextEncoder:
    def __init__(self, params):
        self.params = params
        self.device = get_torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(params.text_model_name)
        self.model = AutoModel.from_pretrained(params.text_model_name).to(self.device)
        self.model.eval()

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        emb = (last_hidden_state * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        return emb

    def fit(self, offers: pd.DataFrame) -> OfferTextArtifacts:
        work = offers[["offer_id", "offer_text"]].copy()
        work["offer_text"] = work["offer_text"].fillna("").astype(str)
        offer_ids = work["offer_id"].astype(int).to_numpy()
        texts = work["offer_text"].tolist()
        all_embs = []
        batch_size = self.params.text_batch_size
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Offer text embeddings"):
                chunk = texts[i : i + batch_size]
                tokens = self.tokenizer(
                    chunk,
                    max_length=self.params.offer_text_max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                out = self.model(**tokens)
                pooled = self._mean_pool(out.last_hidden_state, tokens["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embs.append(pooled.cpu().numpy().astype(np.float32))
        embs = np.vstack(all_embs) if all_embs else np.zeros((0, 768), dtype=np.float32)
        index = faiss.IndexFlatIP(embs.shape[1])
        if len(embs) > 0:
            index.add(embs)
        return OfferTextArtifacts(offer_ids=offer_ids, embeddings=embs, faiss_index=index)


def similar_offers(art: OfferTextArtifacts, offer_id: int, top_k: int) -> list[int]:
    mapping = {oid: idx for idx, oid in enumerate(art.offer_ids.tolist())}
    if offer_id not in mapping or len(art.offer_ids) == 0:
        return []
    idx = mapping[offer_id]
    query = art.embeddings[idx : idx + 1]
    scores, indices = art.faiss_index.search(query, top_k + 1)
    out = []
    for j in indices[0].tolist():
        if j < 0:
            continue
        oid = int(art.offer_ids[j])
        if oid != offer_id:
            out.append(oid)
        if len(out) >= top_k:
            break
    return out
