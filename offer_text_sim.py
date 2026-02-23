from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class OfferSimArgs:
    model_dir: str
    k_offersim: int = 50
    batch_size: int = 128
    max_length: int = 256


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-12)
    return summed / denom


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    out_chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts", unit="batch"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        emb = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out_chunks.append(emb.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.vstack(out_chunks) if out_chunks else np.zeros((0, 0), dtype=np.float32)


def find_similar_offers(
    offer_v1: pd.DataFrame,
    target_text: str,
    args: OfferSimArgs,
) -> pd.DataFrame:
    if "offer_id" not in offer_v1.columns or "offer_text" not in offer_v1.columns:
        raise ValueError("offer_v1 must contain columns: ['offer_id', 'offer_text']")

    if not isinstance(target_text, str) or len(target_text.strip()) == 0:
        raise ValueError("target_text must be a non-empty string")

    device = _pick_device()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)

    offer_ids = offer_v1["offer_id"].to_numpy()
    offer_texts = offer_v1["offer_text"].fillna("").astype(str).tolist()

    target_emb = encode_texts(
        texts=[target_text],
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=1,
        max_length=args.max_length,
    )[0]

    X = encode_texts(
        texts=offer_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    sims = X @ target_emb.astype(np.float32, copy=False)

    k = int(min(args.k_offersim, sims.shape[0]))
    if k <= 0:
        return pd.DataFrame({"offer_id": [], "similarity": []})

    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx], kind="mergesort")]

    res = pd.DataFrame(
        {
            "offer_id": offer_ids[idx],
            "similarity": sims[idx].astype(float),
        }
    )
    return res.reset_index(drop=True)