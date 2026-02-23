import os
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scipy import sparse

try:
    from catboost import CatBoostRanker, Pool
except Exception as e:
    raise RuntimeError("catboost is required") from e


def _read_csv(path: Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    if parse_dates is None:
        parse_dates = []
    return pd.read_csv(path, low_memory=False, parse_dates=parse_dates)


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_pickle(obj, path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_json(obj, path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_sparse_csr(mat: sparse.csr_matrix, path: Path) -> None:
    _ensure_dir(path.parent)
    sparse.save_npz(str(path), mat)


def _load_sparse_csr(path: Path) -> sparse.csr_matrix:
    return sparse.load_npz(str(path))


def _stable_argsort_desc(x: np.ndarray) -> np.ndarray:
    return np.argsort(-x, kind="mergesort")


def _searchsorted_positions(sorted_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_ids, ids)
    ok = (pos >= 0) & (pos < sorted_ids.shape[0]) & (sorted_ids[pos] == ids)
    return pos[ok]


def _hash_buckets_for_series(s: pd.Series, salt: np.uint64, n_buckets: int) -> np.ndarray:
    ss = s.astype("string").fillna("__nan__")
    hv = pd.util.hash_pandas_object(ss, index=False).to_numpy(dtype=np.uint64)
    hv = hv ^ salt
    return (hv % np.uint64(n_buckets)).astype(np.int32)


@dataclass
class PathsConfig:
    data_dir: str = "data/v1"
    work_dir: str = "work"
    artifacts_dir: str = "artifacts"
    global_features_path: str = "work/features/global_features_made.csv"
    minilm_dir: str = "artifacts/models/minilm_l12_v2"


@dataclass
class CandidateConfig:
    n_hash: int = 1 << 18
    topk_seed: int = 5000
    k_offersim: int = 300
    topk_offersim_users: int = 5000
    k_receipts_cats: int = 10
    topk_receipts_users: int = 5000
    candidates_size: int = 10000
    alpha: float = 1.0
    beta: float = 0.3
    p: float = 3.0
    filter_known_by_transactions_only: bool = False


@dataclass
class TrainConfig:
    max_offers: int = 2000
    val_ratio: float = 0.2
    neg_multiplier: float = 0.25
    seed: int = 42
    iterations: int = 1200
    chunk_iters: int = 100
    lr: float = 0.08
    depth: int = 8
    l2_leaf_reg: float = 5.0
    random_strength: float = 1.0


class MiniLMEmbedder:
    def __init__(self, model_dir: Path, batch_size: int = 128, device: Optional[str] = None):
        self.model_dir = model_dir
        self.batch_size = int(batch_size)
        self.device = device
        self._st = None
        self._tok = None
        self._mdl = None
        self._use_sentence_transformers = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(str(self.model_dir))
            if self.device is not None:
                self._st = self._st.to(self.device)
            self._use_sentence_transformers = True
            return
        except Exception:
            self._use_sentence_transformers = False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._mdl = AutoModel.from_pretrained(str(self.model_dir))
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._mdl = self._mdl.to(self.device)
            self._mdl.eval()
        except Exception as e:
            raise RuntimeError("Failed to load MiniLM model (sentence_transformers or transformers fallback).") from e

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 384), dtype=np.float32)

        if self._use_sentence_transformers:
            emb = self._st.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            ).astype(np.float32)
            return emb

        import torch

        out_chunks = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encode", leave=False):
            batch = texts[i : i + self.batch_size]
            tok = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with torch.no_grad():
                hs = self._mdl(**tok).last_hidden_state
                mask = tok["attention_mask"].unsqueeze(-1).to(hs.dtype)
                pooled = (hs * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
            vec = pooled.detach().cpu().numpy().astype(np.float32)
            out_chunks.append(vec)
        emb = np.vstack(out_chunks)
        if normalize:
            n = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(n, 1e-12)
        return emb


class UserVectors:
    def __init__(self, df_global: pd.DataFrame, n_hash: int, cache_dir: Path):
        self.cache_dir = cache_dir
        _ensure_dir(self.cache_dir)

        self.n_hash = int(n_hash)
        self.user_ids = None
        self.num_cols = None
        self.cat_cols = None
        self.X_num = None
        self.num_mean = None
        self.num_std = None
        self.X_cat = None

        self._build_or_load(df_global)

    def _build_or_load(self, df_global: pd.DataFrame) -> None:
        meta_path = self.cache_dir / "user_vectors_meta.json"
        xnum_path = self.cache_dir / "X_num.npy"
        xcat_path = self.cache_dir / "X_cat.npz"

        if meta_path.exists() and xnum_path.exists() and xcat_path.exists():
            meta = _load_json(meta_path)
            self.user_ids = np.array(meta["user_ids"], dtype=np.int64)
            self.num_cols = meta["num_cols"]
            self.cat_cols = meta["cat_cols"]
            self.num_mean = np.array(meta["num_mean"], dtype=np.float32)
            self.num_std = np.array(meta["num_std"], dtype=np.float32)
            self.X_num = np.load(xnum_path).astype(np.float32)
            self.X_cat = _load_sparse_csr(xcat_path).astype(np.float32)
            return

        df = df_global.copy()
        if "user_id" not in df.columns:
            raise ValueError("global_features must contain user_id")

        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["user_id"])
        df["user_id"] = df["user_id"].astype(np.int64)
        df = df.sort_values("user_id", kind="mergesort").reset_index(drop=True)

        self.user_ids = df["user_id"].to_numpy(dtype=np.int64)

        feat_cols = [c for c in df.columns if c != "user_id"]
        num_cols = []
        cat_cols = []
        for c in feat_cols:
            dt = df[c].dtype
            if pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt):
                num_cols.append(c)
            else:
                cat_cols.append(c)

        self.num_cols = num_cols
        self.cat_cols = cat_cols

        X_num = df[self.num_cols].to_numpy(dtype=np.float32, copy=True) if len(self.num_cols) else np.zeros((df.shape[0], 0), dtype=np.float32)
        if X_num.shape[1] > 0:
            mean = np.nanmean(X_num, axis=0).astype(np.float32)
            std = np.nanstd(X_num, axis=0).astype(np.float32)
            std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
            Xn = (X_num - mean) / std
            Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            mean = np.zeros((0,), dtype=np.float32)
            std = np.ones((0,), dtype=np.float32)
            Xn = X_num

        self.X_num = Xn
        self.num_mean = mean
        self.num_std = std

        n = df.shape[0]
        if len(self.cat_cols) == 0:
            self.X_cat = sparse.csr_matrix((n, self.n_hash), dtype=np.float32)
        else:
            rows = np.repeat(np.arange(n, dtype=np.int32), len(self.cat_cols))
            cols_all = []
            salt0 = np.uint64(0x9e3779b97f4a7c15)
            for j, c in enumerate(tqdm(self.cat_cols, desc="Hash cats")):
                salt = salt0 ^ np.uint64((j + 1) * 0x85ebca6b)
                b = _hash_buckets_for_series(df[c], salt, self.n_hash)
                cols_all.append(b)
            cols = np.concatenate(cols_all, axis=0).astype(np.int32, copy=False)
            data = np.ones_like(cols, dtype=np.float32)
            Xc = sparse.coo_matrix((data, (rows, cols)), shape=(n, self.n_hash), dtype=np.float32)
            self.X_cat = Xc.tocsr()

        meta = {
            "user_ids": self.user_ids.tolist(),
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "num_mean": self.num_mean.tolist(),
            "num_std": self.num_std.tolist(),
        }
        _save_json(meta, meta_path)
        _ensure_dir(xnum_path.parent)
        np.save(xnum_path, self.X_num.astype(np.float32))
        _save_sparse_csr(self.X_cat, xcat_path)

    def centroid(self, seed_user_ids: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
        seed_user_ids = np.asarray(seed_user_ids, dtype=np.int64)
        if seed_user_ids.size == 0:
            q_num = np.zeros((self.X_num.shape[1],), dtype=np.float32)
            q_cat = sparse.csr_matrix((1, self.n_hash), dtype=np.float32)
            return q_num, q_cat

        pos = _searchsorted_positions(self.user_ids, seed_user_ids)
        if pos.size == 0:
            q_num = np.zeros((self.X_num.shape[1],), dtype=np.float32)
            q_cat = sparse.csr_matrix((1, self.n_hash), dtype=np.float32)
            return q_num, q_cat

        q_num = self.X_num[pos].mean(axis=0).astype(np.float32, copy=False)

        m = self.X_cat[pos].mean(axis=0)
        if sparse.issparse(m):
            q_cat = m.tocsr().astype(np.float32)
        else:
            q_cat = sparse.csr_matrix(np.asarray(m, dtype=np.float32))

        n_num = float(np.dot(q_num, q_num)) if q_num.size else 0.0
        n_cat = float(q_cat.multiply(q_cat).sum())
        norm = float(np.sqrt(n_num + n_cat))

        if norm < 1e-12:
            return q_num, q_cat

        inv = np.float32(1.0 / norm)
        q_num = (q_num * inv).astype(np.float32, copy=False)
        q_cat = (q_cat * inv).astype(np.float32)
        return q_num, q_cat

    def topk_similar(
        self,
        q_num: np.ndarray,
        q_cat: sparse.csr_matrix,
        k: int,
        exclude_user_ids: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        k = int(k)
        if k <= 0:
            return pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score": np.array([], dtype=np.float32)})

        s = np.zeros((self.user_ids.shape[0],), dtype=np.float32)
        if self.X_num.shape[1] > 0:
            s += (self.X_num @ q_num.astype(np.float32)).astype(np.float32, copy=False)
        if q_cat is not None:
            s += self.X_cat.dot(q_cat.T).toarray().reshape(-1).astype(np.float32, copy=False)

        if exclude_user_ids is not None and len(exclude_user_ids) > 0:
            exclude_user_ids = np.asarray(exclude_user_ids, dtype=np.int64)
            pos = _searchsorted_positions(self.user_ids, exclude_user_ids)
            if pos.size > 0:
                s[pos] = -1e9

        k_eff = min(k, s.shape[0])
        idx = np.argpartition(-s, k_eff - 1)[:k_eff]
        idx = idx[_stable_argsort_desc(s[idx])]
        return pd.DataFrame({"user_id": self.user_ids[idx], "score": s[idx].astype(np.float32)})


class OfferIndex:
    def __init__(self, offers: pd.DataFrame, embedder: MiniLMEmbedder, cache_dir: Path):
        self.cache_dir = cache_dir
        _ensure_dir(self.cache_dir)

        self.offer_ids = None
        self.start_date = None
        self.end_date = None
        self.offer_text = None
        self.emb = None

        self._build_or_load(offers, embedder)

    def _build_or_load(self, offers: pd.DataFrame, embedder: MiniLMEmbedder) -> None:
        emb_path = self.cache_dir / "offer_emb.npy"
        ids_path = self.cache_dir / "offer_ids.npy"
        meta_path = self.cache_dir / "offer_meta.parquet"

        if emb_path.exists() and ids_path.exists() and meta_path.exists():
            self.emb = np.load(emb_path).astype(np.float32)
            self.offer_ids = np.load(ids_path).astype(np.int64)
            meta = pd.read_parquet(meta_path)
            self.start_date = _to_datetime(meta["start_date"])
            self.end_date = _to_datetime(meta["end_date"])
            self.offer_text = meta["offer_text"].astype("string")
            return

        df = offers.copy()
        df["offer_id"] = pd.to_numeric(df["offer_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["offer_id"])
        df["offer_id"] = df["offer_id"].astype(np.int64)
        df["start_date"] = _to_datetime(df["start_date"])
        df["end_date"] = _to_datetime(df["end_date"])
        if "offer_text" not in df.columns:
            df["offer_text"] = ""
        df["offer_text"] = df["offer_text"].astype("string").fillna("")

        df = df.sort_values("offer_id", kind="mergesort").reset_index(drop=True)

        self.offer_ids = df["offer_id"].to_numpy(dtype=np.int64)
        self.start_date = df["start_date"]
        self.end_date = df["end_date"]
        self.offer_text = df["offer_text"]

        texts = self.offer_text.tolist()
        self.emb = embedder.encode(texts, normalize=True).astype(np.float32)

        _ensure_dir(emb_path.parent)
        np.save(emb_path, self.emb)
        np.save(ids_path, self.offer_ids)
        df[["offer_id", "start_date", "end_date", "offer_text"]].to_parquet(meta_path, index=False)

    def get_offer_row_idx(self, offer_id: int) -> int:
        offer_id = int(offer_id)
        pos = np.searchsorted(self.offer_ids, offer_id)
        if pos < 0 or pos >= self.offer_ids.shape[0] or self.offer_ids[pos] != offer_id:
            return -1
        return int(pos)

    def offer_embedding(self, offer_id: int) -> Optional[np.ndarray]:
        i = self.get_offer_row_idx(offer_id)
        if i < 0:
            return None
        return self.emb[i]

    def topk_similar_offers(self, offer_id: int, k: int) -> np.ndarray:
        i = self.get_offer_row_idx(offer_id)
        if i < 0:
            return np.array([], dtype=np.int64)
        v = self.emb[i]
        sims = (self.emb @ v).astype(np.float32, copy=False)
        sims[i] = -1e9
        k_eff = min(int(k), sims.shape[0] - 1)
        if k_eff <= 0:
            return np.array([], dtype=np.int64)
        idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[_stable_argsort_desc(sims[idx])]
        return self.offer_ids[idx].astype(np.int64)


class CategoryIndex:
    def __init__(self, receipts: pd.DataFrame, embedder: MiniLMEmbedder, cache_dir: Path):
        self.cache_dir = cache_dir
        _ensure_dir(self.cache_dir)

        self.cats = None
        self.emb = None

        self._build_or_load(receipts, embedder)

    def _build_or_load(self, receipts: pd.DataFrame, embedder: MiniLMEmbedder) -> None:
        emb_path = self.cache_dir / "cat_emb.npy"
        cats_path = self.cache_dir / "cats.npy"

        if emb_path.exists() and cats_path.exists():
            self.emb = np.load(emb_path).astype(np.float32)
            self.cats = np.load(cats_path).astype("U")
            return

        if receipts is None or receipts.shape[0] == 0:
            self.cats = np.array([], dtype="U")
            self.emb = np.zeros((0, 384), dtype=np.float32)
            np.save(emb_path, self.emb)
            np.save(cats_path, self.cats)
            return

        cats = receipts["category_name"].astype("string").fillna("").unique()
        cats = np.array(sorted([c for c in cats if c is not None and len(str(c)) > 0]), dtype="U")
        self.cats = cats
        self.emb = embedder.encode(cats.tolist(), normalize=True).astype(np.float32)
        np.save(emb_path, self.emb)
        np.save(cats_path, self.cats)

    def topk_for_offer(self, offer_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.emb.shape[0] == 0:
            return np.array([], dtype="U"), np.array([], dtype=np.float32)
        sims = (self.emb @ offer_emb).astype(np.float32, copy=False)
        k_eff = min(int(k), sims.shape[0])
        idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[_stable_argsort_desc(sims[idx])]
        return self.cats[idx], sims[idx].astype(np.float32)


class CandidateBuilder:
    def __init__(
        self,
        paths: PathsConfig,
        cand_cfg: CandidateConfig,
        user_vectors: UserVectors,
        offer_index: OfferIndex,
        cat_index: CategoryIndex,
        df_offer: pd.DataFrame,
        df_merchant: pd.DataFrame,
        df_tx: pd.DataFrame,
        df_seen: pd.DataFrame,
        df_act: pd.DataFrame,
        df_rew: pd.DataFrame,
        df_receipts: pd.DataFrame,
    ):
        self.paths = paths
        self.cfg = cand_cfg
        self.U = user_vectors
        self.offer_index = offer_index
        self.cat_index = cat_index

        self.offer = df_offer
        self.merchant = df_merchant
        self.tx = df_tx
        self.seen = df_seen
        self.act = df_act
        self.rew = df_rew
        self.receipts = df_receipts

        self.offer = self.offer.copy()
        self.offer["offer_id"] = pd.to_numeric(self.offer["offer_id"], errors="coerce").astype("Int64")
        self.offer["merchant_id_offer"] = pd.to_numeric(self.offer["merchant_id_offer"], errors="coerce").astype("Int64")
        self.offer["start_date"] = _to_datetime(self.offer["start_date"])
        self.offer["end_date"] = _to_datetime(self.offer["end_date"])
        self.offer = self.offer.dropna(subset=["offer_id", "merchant_id_offer"]).copy()
        self.offer["offer_id"] = self.offer["offer_id"].astype(np.int64)
        self.offer["merchant_id_offer"] = self.offer["merchant_id_offer"].astype(np.int64)

        self.merchant = self.merchant.copy()
        self.merchant["merchant_id_offer"] = pd.to_numeric(self.merchant["merchant_id_offer"], errors="coerce").astype("Int64")
        self.merchant["brand_dk"] = pd.to_numeric(self.merchant["brand_dk"], errors="coerce").astype("Int64")
        self.merchant = self.merchant.dropna(subset=["merchant_id_offer", "brand_dk"]).copy()
        self.merchant["merchant_id_offer"] = self.merchant["merchant_id_offer"].astype(np.int64)
        self.merchant["brand_dk"] = self.merchant["brand_dk"].astype(np.int64)

        self.tx = self.tx.copy()
        self.tx["user_id"] = pd.to_numeric(self.tx["user_id"], errors="coerce").astype("Int64")
        self.tx["brand_dk"] = pd.to_numeric(self.tx["brand_dk"], errors="coerce").astype("Int64")
        self.tx["event_date"] = _to_datetime(self.tx["event_date"])
        self.tx = self.tx.dropna(subset=["user_id", "brand_dk", "event_date"]).copy()
        self.tx["user_id"] = self.tx["user_id"].astype(np.int64)
        self.tx["brand_dk"] = self.tx["brand_dk"].astype(np.int64)

        for df, dc, dcol in [
            (self.seen, "offer_id", None),
            (self.act, "offer_id", "activation_date"),
            (self.rew, "offer_id", "event_date"),
        ]:
            if df is None or df.shape[0] == 0:
                continue
            df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
            df[dc] = pd.to_numeric(df[dc], errors="coerce").astype("Int64")
            if dcol is not None and dcol in df.columns:
                df[dcol] = _to_datetime(df[dcol])
            df.dropna(subset=["user_id", dc], inplace=True)
            df["user_id"] = df["user_id"].astype(np.int64)
            df[dc] = df[dc].astype(np.int64)

        if self.receipts is not None and self.receipts.shape[0] > 0:
            self.receipts = self.receipts.copy()
            self.receipts["user_id"] = pd.to_numeric(self.receipts["user_id"], errors="coerce").astype("Int64")
            self.receipts["date_operated"] = _to_datetime(self.receipts["date_operated"])
            self.receipts["items_count"] = pd.to_numeric(self.receipts.get("items_count", 1), errors="coerce")
            self.receipts["category_name"] = self.receipts["category_name"].astype("string").fillna("")
            self.receipts = self.receipts.dropna(subset=["user_id", "date_operated"]).copy()
            self.receipts["user_id"] = self.receipts["user_id"].astype(np.int64)

        self._offer2merchant = None
        self._merchant2brand = None
        self._init_maps()

    def _init_maps(self) -> None:
        self._offer2merchant = self.offer.set_index("offer_id")["merchant_id_offer"]
        self._merchant2brand = self.merchant.set_index("merchant_id_offer")["brand_dk"]

    def offer_context(self, offer_id: int) -> tuple[int, int, pd.Timestamp, pd.Timestamp]:
        offer_id = int(offer_id)
        if offer_id not in self._offer2merchant.index:
            raise KeyError(f"offer_id {offer_id} not found")
        m_id = int(self._offer2merchant.loc[offer_id])
        if m_id not in self._merchant2brand.index:
            raise KeyError(f"merchant_id_offer {m_id} not found")
        b = int(self._merchant2brand.loc[m_id])
        row = self.offer[self.offer["offer_id"] == offer_id]
        t0 = row["start_date"].iloc[0]
        t1 = row["end_date"].iloc[0]
        return m_id, b, t0, t1

    def known_users(self, brand_dk: int, t0: pd.Timestamp, offer_id: int) -> np.ndarray:
        brand_dk = int(brand_dk)
        tx_known = self.tx.loc[(self.tx["brand_dk"] == brand_dk) & (self.tx["event_date"] < t0), "user_id"].unique()
        if self.cfg.filter_known_by_transactions_only:
            return np.asarray(tx_known, dtype=np.int64)

        seen_known = np.array([], dtype=np.int64)
        act_known = np.array([], dtype=np.int64)
        rew_known = np.array([], dtype=np.int64)

        if self.seen is not None and self.seen.shape[0] > 0 and "start_date" in self.seen.columns:
            s = self.seen.copy()
            s["start_date"] = _to_datetime(s["start_date"])
            seen_known = s.loc[(s["offer_id"] == offer_id) & (s["start_date"] < t0), "user_id"].unique()

        if self.act is not None and self.act.shape[0] > 0 and "activation_date" in self.act.columns:
            act_known = self.act.loc[(self.act["offer_id"] == offer_id) & (self.act["activation_date"] < t0), "user_id"].unique()

        if self.rew is not None and self.rew.shape[0] > 0 and "event_date" in self.rew.columns:
            rew_known = self.rew.loc[(self.rew["offer_id"] == offer_id) & (self.rew["event_date"] < t0), "user_id"].unique()

        out = np.unique(np.concatenate([tx_known, seen_known, act_known, rew_known]).astype(np.int64, copy=False))
        return out

    def seed_positives(self, brand_dk: int, t0: pd.Timestamp, offer_id: int) -> np.ndarray:
        brand_dk = int(brand_dk)
        tx_pos = self.tx.loc[(self.tx["brand_dk"] == brand_dk) & (self.tx["event_date"] < t0), "user_id"].unique()

        act_pos = np.array([], dtype=np.int64)
        rew_pos = np.array([], dtype=np.int64)

        if self.act is not None and self.act.shape[0] > 0 and "activation_date" in self.act.columns:
            act_pos = self.act.loc[(self.act["offer_id"] == offer_id) & (self.act["activation_date"] < t0), "user_id"].unique()

        if self.rew is not None and self.rew.shape[0] > 0 and "event_date" in self.rew.columns:
            rew_pos = self.rew.loc[(self.rew["offer_id"] == offer_id) & (self.rew["event_date"] < t0), "user_id"].unique()

        out = np.unique(np.concatenate([tx_pos, act_pos, rew_pos]).astype(np.int64, copy=False))
        return out

    def candidates_seed_centroid(self, offer_id: int) -> pd.DataFrame:
        cache_path = Path(self.paths.work_dir) / "cache" / "candidates" / f"offer_{offer_id}_seed.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        _, brand, t0, _ = self.offer_context(offer_id)
        known = self.known_users(brand, t0, offer_id)
        pos = self.seed_positives(brand, t0, offer_id)
        q_num, q_cat = self.U.centroid(pos)
        cand = self.U.topk_similar(q_num, q_cat, self.cfg.topk_seed, exclude_user_ids=known)
        cand = cand.rename(columns={"score": "score_seed"})
        cand.to_parquet(cache_path, index=False)
        return cand

    def candidates_similar_offers_centroid(self, offer_id: int) -> pd.DataFrame:
        cache_path = Path(self.paths.work_dir) / "cache" / "candidates" / f"offer_{offer_id}_offersim.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        _, brand, t0, _ = self.offer_context(offer_id)
        known = self.known_users(brand, t0, offer_id)

        sim_offers = self.offer_index.topk_similar_offers(offer_id, self.cfg.k_offersim * 3)
        if sim_offers.size == 0:
            out = pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_offersim": np.array([], dtype=np.float32)})
            out.to_parquet(cache_path, index=False)
            return out

        idx = np.searchsorted(self.offer_index.offer_ids, sim_offers)
        ok = (idx >= 0) & (idx < self.offer_index.offer_ids.shape[0]) & (self.offer_index.offer_ids[idx] == sim_offers)
        idx = idx[ok]
        valid = sim_offers[ok]
        sd = self.offer_index.start_date.iloc[idx].to_numpy()
        valid = valid[pd.to_datetime(sd) < t0]

        if valid.size == 0:
            out = pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_offersim": np.array([], dtype=np.float32)})
            out.to_parquet(cache_path, index=False)
            return out

        pos_all = []
        for oid in valid[: self.cfg.k_offersim]:
            pos = self.seed_positives(brand, t0, int(oid))
            if pos.size > 0:
                pos_all.append(pos)
        if len(pos_all) == 0:
            out = pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_offersim": np.array([], dtype=np.float32)})
            out.to_parquet(cache_path, index=False)
            return out

        pos_u = np.unique(np.concatenate(pos_all).astype(np.int64, copy=False))
        q_num, q_cat = self.U.centroid(pos_u)
        cand = self.U.topk_similar(q_num, q_cat, self.cfg.topk_offersim_users, exclude_user_ids=known)
        cand = cand.rename(columns={"score": "score_offersim"})
        cand.to_parquet(cache_path, index=False)
        return cand

    def candidates_receipts(self, offer_id: int) -> pd.DataFrame:
        cache_path = Path(self.paths.work_dir) / "cache" / "candidates" / f"offer_{offer_id}_receipts.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        if self.receipts is None or self.receipts.shape[0] == 0:
            out = pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_receipts": np.array([], dtype=np.float32)})
            out.to_parquet(cache_path, index=False)
            return out

        _, brand, t0, _ = self.offer_context(offer_id)
        known = self.known_users(brand, t0, offer_id)

        offer_emb = self.offer_index.offer_embedding(offer_id)
        if offer_emb is None:
            out = pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_receipts": np.array([], dtype=np.float32)})
            out.to_parquet(cache_path, index=False)
            return out

        top_cats, sims = self.cat_index.topk_for_offer(offer_emb, self.cfg.k_receipts_cats)
        if top_cats.size == 0:
            out = pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_receipts": np.array([], dtype=np.float32)})
            out.to_parquet(cache_path, index=False)
            return out

        r = self.receipts
        t45 = t0 - pd.Timedelta(days=45)
        t10 = t0 - pd.Timedelta(days=10)

        r45 = r[(r["date_operated"] >= t45) & (r["date_operated"] < t0)]
        if r45.shape[0] == 0:
            out = pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_receipts": np.array([], dtype=np.float32)})
            out.to_parquet(cache_path, index=False)
            return out

        r45 = r45.copy()
        ic = r45["items_count"].to_numpy(dtype=np.float32, copy=False)
        ic = np.where(np.isfinite(ic), ic, 1.0).astype(np.float32, copy=False)
        r45["ic"] = ic
        r45["day"] = r45["date_operated"].dt.date

        active_days = r45.groupby("user_id", sort=False)["day"].nunique().astype(np.float32)
        norm = (1.0 / np.sqrt(1.0 + active_days)).astype(np.float32)

        g45 = r45.groupby(["user_id", "category_name"], sort=False)["ic"].sum().astype(np.float32)
        df_c = g45.reset_index().groupby("category_name", sort=False)["user_id"].nunique().astype(np.float32)
        N = float(active_days.shape[0])

        r10 = r45[r45["date_operated"] >= t10]
        g10 = r10.groupby(["user_id", "category_name"], sort=False)["ic"].sum().astype(np.float32)

        cats_set = set(top_cats.tolist())
        g45 = g45[g45.index.get_level_values(1).isin(cats_set)]
        g10 = g10[g10.index.get_level_values(1).isin(cats_set)]

        df_top = df_c.reindex(top_cats).fillna(0.0).to_numpy(dtype=np.float32)
        sim_pos = np.maximum(sims.astype(np.float32), 0.0)
        q = np.power(sim_pos, np.float32(self.cfg.p)).astype(np.float32)
        idf = np.log1p(N / (1.0 + df_top)).astype(np.float32)
        w = (q * idf).astype(np.float32)

        w_map = pd.Series(w, index=pd.Index(top_cats, dtype="string"))

        a = np.float32(self.cfg.alpha)
        b = np.float32(self.cfg.beta)

        g45_df = g45.reset_index(name="cnt45")
        g10_df = g10.reset_index(name="cnt10")
        m = g45_df.merge(g10_df, on=["user_id", "category_name"], how="left")
        m["cnt10"] = m["cnt10"].fillna(0.0).astype(np.float32)
        m["w"] = m["category_name"].map(w_map).astype(np.float32)

        tf10 = np.log1p(m["cnt10"].to_numpy(dtype=np.float32, copy=False))
        tf45 = np.log1p(m["cnt45"].to_numpy(dtype=np.float32, copy=False))
        ww = m["w"].to_numpy(dtype=np.float32, copy=False)
        contrib = ww * (a * tf10 + b * tf45)
        m["contrib"] = contrib.astype(np.float32)

        score = m.groupby("user_id", sort=False)["contrib"].sum().astype(np.float32)
        score = score.mul(norm, fill_value=0.0).astype(np.float32)
        score = score[~score.index.isin(known)]

        if score.shape[0] == 0:
            out = pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_receipts": np.array([], dtype=np.float32)})
            out.to_parquet(cache_path, index=False)
            return out

        k_eff = min(int(self.cfg.topk_receipts_users), int(score.shape[0]))
        top_idx = np.argpartition(-score.to_numpy(), k_eff - 1)[:k_eff]
        top_u = score.index.to_numpy(dtype=np.int64)[top_idx]
        top_s = score.to_numpy(dtype=np.float32)[top_idx]
        ord2 = _stable_argsort_desc(top_s)
        out = pd.DataFrame({"user_id": top_u[ord2], "score_receipts": top_s[ord2].astype(np.float32)})
        out.to_parquet(cache_path, index=False)
        return out

    def candidates_activity_fill(self, already: np.ndarray, k: int, global_df: pd.DataFrame) -> pd.DataFrame:
        already = np.asarray(already, dtype=np.int64)
        k = int(k)
        if k <= 0:
            return pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_pop": np.array([], dtype=np.float32)})

        cols = list(global_df.columns)
        cand_col = None
        for c in ["tx_count_30d", "active_days_tx_30d", "tx_count_60d", "tx_count_90d", "last_activity_day"]:
            if c in cols:
                cand_col = c
                break
        if cand_col is None:
            cand_col = cols[1] if len(cols) > 1 else None
        if cand_col is None:
            return pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_pop": np.array([], dtype=np.float32)})

        s = pd.to_numeric(global_df[cand_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        uid = pd.to_numeric(global_df["user_id"], errors="coerce").astype("Int64")
        msk = uid.notna().to_numpy()
        uid = uid[msk].astype(np.int64).to_numpy()
        s = s[msk]

        if already.size > 0:
            already_set = set(already.tolist())
            keep = np.array([u not in already_set for u in uid], dtype=bool)
            uid = uid[keep]
            s = s[keep]

        if uid.size == 0:
            return pd.DataFrame({"user_id": np.array([], dtype=np.int64), "score_pop": np.array([], dtype=np.float32)})

        k_eff = min(k, uid.shape[0])
        idx = np.argpartition(-s, k_eff - 1)[:k_eff]
        idx = idx[_stable_argsort_desc(s[idx])]
        return pd.DataFrame({"user_id": uid[idx], "score_pop": s[idx].astype(np.float32)})

    def assemble_candidates(self, offer_id: int, global_df: pd.DataFrame) -> pd.DataFrame:
        c1 = self.candidates_seed_centroid(offer_id)
        c2 = self.candidates_similar_offers_centroid(offer_id)
        c3 = self.candidates_receipts(offer_id)

        df = c1.merge(c2, on="user_id", how="outer").merge(c3, on="user_id", how="outer")
        df["sources_cnt"] = (
            df[["score_seed", "score_offersim", "score_receipts"]]
            .notna()
            .sum(axis=1)
            .astype(np.int16)
        )
        df["score_seed"] = df["score_seed"].fillna(0.0).astype(np.float32)
        df["score_offersim"] = df["score_offersim"].fillna(0.0).astype(np.float32)
        df["score_receipts"] = df["score_receipts"].fillna(0.0).astype(np.float32)

        need = self.cfg.candidates_size - int(df.shape[0])
        if need > 0:
            add = self.candidates_activity_fill(df["user_id"].to_numpy(dtype=np.int64), need, global_df)
            df = df.merge(add, on="user_id", how="outer")
        else:
            df["score_pop"] = np.float32(0.0)

        df["score_pop"] = df["score_pop"].fillna(0.0).astype(np.float32)
        df = df.sort_values(["sources_cnt", "score_seed", "score_offersim", "score_receipts", "score_pop"], ascending=False, kind="mergesort")
        df = df.head(self.cfg.candidates_size).reset_index(drop=True)
        return df


def build_label_transaction_new_clients(
    tx: pd.DataFrame,
    brand_dk: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    t = tx[(tx["brand_dk"] == int(brand_dk)) & tx["event_date"].notna()]
    before = t[t["event_date"] < start]["user_id"].unique().astype(np.int64, copy=False)
    during = t[(t["event_date"] >= start) & (t["event_date"] < end)]["user_id"].unique().astype(np.int64, copy=False)
    before = np.unique(before)
    during = np.unique(during)
    y = np.setdiff1d(during, before, assume_unique=False).astype(np.int64, copy=False)
    return y


def map_at_100(df: pd.DataFrame, group_col: str, y_col: str, pred_col: str) -> float:
    vals = []
    for gid, g in tqdm(df.groupby(group_col, sort=False), desc="MAP@100", leave=False):
        y = g[y_col].to_numpy(dtype=np.int8, copy=False)
        p = g[pred_col].to_numpy(dtype=np.float32, copy=False)
        if y.sum() == 0:
            vals.append(0.0)
            continue
        order = _stable_argsort_desc(p)
        y_sorted = y[order]
        k = min(100, y_sorted.shape[0])
        yk = y_sorted[:k]
        denom = float(min(int(y.sum()), 100))
        if denom <= 0:
            vals.append(0.0)
            continue
        cum = np.cumsum(yk, dtype=np.int32)
        pos_idx = np.flatnonzero(yk == 1)
        if pos_idx.size == 0:
            vals.append(0.0)
            continue
        ap = (cum[pos_idx] / (pos_idx + 1)).sum() / denom
        vals.append(float(ap))
    return float(np.mean(vals)) if len(vals) else 0.0


def train_catboost_ranker_with_tqdm(
    train_pool: Pool,
    val_pool: Pool,
    params: dict,
    iterations: int,
    chunk_iters: int,
) -> CatBoostRanker:
    iterations = int(iterations)
    chunk_iters = int(chunk_iters)
    model = CatBoostRanker(**{k: v for k, v in params.items() if k != "iterations"})
    pbar = tqdm(total=iterations, desc="CatBoost train", leave=True)
    trained = 0
    init_model = None
    while trained < iterations:
        it = min(chunk_iters, iterations - trained)
        model.set_params(iterations=it)
        model.fit(
            train_pool,
            eval_set=val_pool,
            init_model=init_model,
        )
        init_model = model
        trained += it
        pbar.update(it)
    pbar.close()
    return model


def main_train(paths: PathsConfig, cand_cfg: CandidateConfig, tr_cfg: TrainConfig) -> None:
    np.random.seed(tr_cfg.seed)

    data_dir = Path(paths.data_dir)
    work_dir = Path(paths.work_dir)
    art_dir = Path(paths.artifacts_dir)

    df_global = _read_csv(Path(paths.global_features_path))
    df_offer = _read_csv(data_dir / "t_offer.csv", parse_dates=["start_date", "end_date"])
    df_merchant = _read_csv(data_dir / "t_merchant.csv")
    df_tx = _read_csv(data_dir / "prod_financial_transaction.csv", parse_dates=["event_date"])
    df_seen = _read_csv(data_dir / "offer_seens.csv", parse_dates=["start_date", "end_date"]) if (data_dir / "offer_seens.csv").exists() else pd.DataFrame()
    df_act = _read_csv(data_dir / "offer_activation.csv", parse_dates=["activation_date"]) if (data_dir / "offer_activation.csv").exists() else pd.DataFrame()
    df_rew = _read_csv(data_dir / "offer_reward.csv", parse_dates=["event_date"]) if (data_dir / "offer_reward.csv").exists() else pd.DataFrame()
    df_receipts = _read_csv(data_dir / "receipts.csv", parse_dates=["date_operated"]) if (data_dir / "receipts.csv").exists() else pd.DataFrame()

    embedder = MiniLMEmbedder(Path(paths.minilm_dir), batch_size=64)
    user_vectors = UserVectors(df_global, cand_cfg.n_hash, work_dir / "cache" / "user_vectors")
    offer_index = OfferIndex(df_offer, embedder, work_dir / "cache" / "text" / "offers")
    cat_index = CategoryIndex(df_receipts, embedder, work_dir / "cache" / "text" / "categories")

    builder = CandidateBuilder(
        paths=paths,
        cand_cfg=cand_cfg,
        user_vectors=user_vectors,
        offer_index=offer_index,
        cat_index=cat_index,
        df_offer=df_offer,
        df_merchant=df_merchant,
        df_tx=df_tx,
        df_seen=df_seen,
        df_act=df_act,
        df_rew=df_rew,
        df_receipts=df_receipts,
    )

    offers = df_offer.copy()
    offers["offer_id"] = pd.to_numeric(offers["offer_id"], errors="coerce").astype("Int64")
    offers = offers.dropna(subset=["offer_id", "start_date", "end_date"]).copy()
    offers["offer_id"] = offers["offer_id"].astype(np.int64)
    offers["start_date"] = _to_datetime(offers["start_date"])
    offers["end_date"] = _to_datetime(offers["end_date"])
    offers = offers.sort_values("start_date", kind="mergesort").reset_index(drop=True)
    if offers.shape[0] > tr_cfg.max_offers:
        offers = offers.iloc[-tr_cfg.max_offers :].reset_index(drop=True)

    n_val = max(1, int(np.floor(offers.shape[0] * tr_cfg.val_ratio)))
    val_offers = offers.iloc[-n_val:]["offer_id"].to_numpy(dtype=np.int64)
    train_offers = offers.iloc[:-n_val]["offer_id"].to_numpy(dtype=np.int64)

    global_df = df_global.copy()
    global_df["user_id"] = pd.to_numeric(global_df["user_id"], errors="coerce").astype("Int64")
    global_df = global_df.dropna(subset=["user_id"]).copy()
    global_df["user_id"] = global_df["user_id"].astype(np.int64)

    train_rows = []
    val_rows = []

    def build_rows_for_offer(offer_id: int) -> Optional[pd.DataFrame]:
        try:
            m_id, brand, t0, t1 = builder.offer_context(int(offer_id))
        except Exception:
            return None

        cand = builder.assemble_candidates(int(offer_id), global_df)
        y_pos = build_label_transaction_new_clients(df_tx, brand, t0, t1)
        y_pos_set = set(y_pos.tolist())

        base = cand.merge(global_df, on="user_id", how="left")
        uids = base["user_id"].to_numpy(dtype=np.int64, copy=False)
        pos = y_pos.astype(np.int64, copy=False)
        base["label"] = np.isin(uids, pos).astype(np.int8)
        base["group_id"] = np.int64(offer_id)
        base["brand_dk"] = np.int64(brand)
        base["merchant_id_offer"] = np.int64(m_id)

        if tr_cfg.neg_multiplier > 0:
            k_neg = int(np.floor(cand.shape[0] * tr_cfg.neg_multiplier))
            if k_neg > 0:
                all_users = global_df["user_id"].to_numpy(dtype=np.int64)
                rng = np.random.default_rng(tr_cfg.seed + int(offer_id))
                neg_u = rng.choice(all_users, size=min(k_neg, all_users.shape[0]), replace=False).astype(np.int64)
                neg_u = np.array([u for u in neg_u if u not in y_pos_set], dtype=np.int64)
                neg_df = pd.DataFrame({"user_id": neg_u})
                neg_df["score_seed"] = np.float32(0.0)
                neg_df["score_offersim"] = np.float32(0.0)
                neg_df["score_receipts"] = np.float32(0.0)
                neg_df["score_pop"] = np.float32(0.0)
                neg_df["sources_cnt"] = np.int16(0)
                neg_df = neg_df.merge(global_df, on="user_id", how="left")
                neg_df["label"] = np.int8(0)
                neg_df["group_id"] = np.int64(offer_id)
                neg_df["brand_dk"] = np.int64(brand)
                neg_df["merchant_id_offer"] = np.int64(m_id)
                base = pd.concat([base, neg_df], axis=0, ignore_index=True)

        return base

    for oid in tqdm(train_offers, desc="Build train"):
        df_o = build_rows_for_offer(int(oid))
        if df_o is None:
            continue
        if int(df_o["label"].sum()) == 0:
            continue
        train_rows.append(df_o)

    for oid in tqdm(val_offers, desc="Build val"):
        df_o = build_rows_for_offer(int(oid))
        if df_o is None:
            continue
        if int(df_o["label"].sum()) == 0:
            continue
        val_rows.append(df_o)

    train_df = pd.concat(train_rows, axis=0, ignore_index=True) if len(train_rows) else pd.DataFrame()
    val_df = pd.concat(val_rows, axis=0, ignore_index=True) if len(val_rows) else pd.DataFrame()
    if train_df.shape[0] == 0 or val_df.shape[0] == 0:
        raise RuntimeError("Empty train/val after filtering. Increase max_offers or relax constraints.")

    drop_cols = set(["label", "group_id"])
    feat_cols = [c for c in train_df.columns if c not in drop_cols]

    for c in feat_cols:
        if pd.api.types.is_bool_dtype(train_df[c].dtype):
            train_df[c] = train_df[c].astype(np.int8)
        if pd.api.types.is_bool_dtype(val_df[c].dtype):
            val_df[c] = val_df[c].astype(np.int8)

    cat_cols = []
    for c in feat_cols:
        dt = train_df[c].dtype
        if (pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt) or pd.api.types.is_categorical_dtype(dt)) and c != "user_id":
            cat_cols.append(c)

    train_pool = Pool(
        train_df[feat_cols],
        label=train_df["label"].to_numpy(dtype=np.float32),
        group_id=train_df["group_id"].to_numpy(dtype=np.int64),
        cat_features=cat_cols,
    )
    val_pool = Pool(
        val_df[feat_cols],
        label=val_df["label"].to_numpy(dtype=np.float32),
        group_id=val_df["group_id"].to_numpy(dtype=np.int64),
        cat_features=cat_cols,
    )

    params = dict(
        loss_function="YetiRank",
        eval_metric="MAP:top=100",
        learning_rate=float(tr_cfg.lr),
        depth=int(tr_cfg.depth),
        l2_leaf_reg=float(tr_cfg.l2_leaf_reg),
        random_strength=float(tr_cfg.random_strength),
        random_seed=int(tr_cfg.seed),
        task_type="CPU",
        iterations=int(tr_cfg.iterations),
        allow_writing_files=True,
        verbose=200,
    )

    model = train_catboost_ranker_with_tqdm(
        train_pool=train_pool,
        val_pool=val_pool,
        params=params,
        iterations=tr_cfg.iterations,
        chunk_iters=tr_cfg.chunk_iters,
    )

    val_pred = model.predict(val_pool).astype(np.float32)
    val_out = val_df[["group_id", "label"]].copy()
    val_out["pred"] = val_pred
    m = map_at_100(val_out, group_col="group_id", y_col="label", pred_col="pred")

    out_dir = work_dir / "models"
    _ensure_dir(out_dir)
    model_path = out_dir / "catboost_ranker.cbm"
    model.save_model(str(model_path))

    meta = {
        "feat_cols": feat_cols,
        "cat_cols": cat_cols,
        "map_at_100_val": m,
        "train_rows": int(train_df.shape[0]),
        "val_rows": int(val_df.shape[0]),
        "train_groups": int(train_df["group_id"].nunique()),
        "val_groups": int(val_df["group_id"].nunique()),
        "params": params,
    }
    _save_json(meta, out_dir / "model_meta.json")

    print(f"Saved model to: {model_path}")
    print(f"VAL MAP@100: {m:.6f}")


if __name__ == "__main__":
    paths = PathsConfig()
    cand_cfg = CandidateConfig()
    tr_cfg = TrainConfig()
    main_train(paths, cand_cfg, tr_cfg)