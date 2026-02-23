from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from tqdm import tqdm


@dataclass(frozen=True)
class HashCentroidConfig:
    n_features: int = 2**20
    batch_size: int = 4096
    alternate_sign: bool = False
    normalize_numeric: bool = True
    clip_z: float = 8.0


def _infer_cols(df: pd.DataFrame, exclude: Sequence[str]) -> tuple[list[str], list[str]]:
    exclude_set = set(exclude)
    num_cols = []
    cat_cols = []
    for c in df.columns:
        if c in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def _fit_numeric_scaler(df: pd.DataFrame, num_cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    if not num_cols:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    x = df[list(num_cols)].to_numpy(dtype=np.float32, copy=False)
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma = np.where(sigma > 1e-6, sigma, 1.0).astype(np.float32)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    return mu, sigma


def _row_to_feature_dict(
    row: pd.Series,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    normalize_numeric: bool,
    clip_z: float,
) -> dict[str, float]:
    d: dict[str, float] = {}
    if num_cols:
        vals = row[list(num_cols)].to_numpy(dtype=np.float32, copy=False)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        if normalize_numeric:
            z = (vals - mu) / sigma
            z = np.clip(z, -clip_z, clip_z)
            for c, v in zip(num_cols, z.tolist()):
                if v != 0.0:
                    d[f"n:{c}"] = float(v)
        else:
            for c, v in zip(num_cols, vals.tolist()):
                if v != 0.0:
                    d[f"n:{c}"] = float(v)

    for c in cat_cols:
        v = row[c]
        if pd.isna(v):
            continue
        s = str(v)
        if s:
            d[f"c:{c}={s}"] = 1.0
    return d


def build_hashed_matrix(
    df: pd.DataFrame,
    user_ids: Sequence[int] | np.ndarray,
    user_id_col: str = "user_id",
    num_cols: Sequence[str] | None = None,
    cat_cols: Sequence[str] | None = None,
    cfg: HashCentroidConfig = HashCentroidConfig(),
) -> tuple[sparse.csr_matrix, np.ndarray, list[str], list[str]]:
    user_ids_arr = np.asarray(user_ids, dtype=np.int64)
    sub = df.loc[df[user_id_col].isin(user_ids_arr)].copy()
    sub[user_id_col] = sub[user_id_col].astype(np.int64)

    sub = sub.set_index(user_id_col).loc[user_ids_arr]
    if num_cols is None or cat_cols is None:
        ncols, ccols = _infer_cols(sub.reset_index(drop=True), exclude=[user_id_col])
        if num_cols is None:
            num_cols = ncols
        if cat_cols is None:
            cat_cols = ccols

    mu, sigma = _fit_numeric_scaler(sub.reset_index(drop=True), num_cols) if cfg.normalize_numeric else (np.array([], dtype=np.float32), np.array([], dtype=np.float32))

    hasher = FeatureHasher(
        n_features=int(cfg.n_features),
        input_type="dict",
        alternate_sign=bool(cfg.alternate_sign),
    )

    batches = []
    n = sub.shape[0]
    for start in tqdm(range(0, n, cfg.batch_size), desc="Hashing users", unit="batch"):
        end = min(start + cfg.batch_size, n)
        block = sub.iloc[start:end]
        dicts = [
            _row_to_feature_dict(r, num_cols, cat_cols, mu, sigma, cfg.normalize_numeric, cfg.clip_z)
            for _, r in block.iterrows()
        ]
        xb = hasher.transform(dicts).tocsr()
        batches.append(xb)

    X = sparse.vstack(batches, format="csr")
    return X, user_ids_arr, list(num_cols), list(cat_cols)


def topk_similar_to_centroid(
    X: sparse.csr_matrix,
    all_user_ids: np.ndarray,
    user_common: Sequence[int] | np.ndarray,
    top_k: int,
    sim_threshold: float = 0.0,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    common = np.asarray(user_common, dtype=np.int64)
    common_mask = np.isin(all_user_ids, common)
    if not np.any(common_mask):
        return pd.DataFrame({"user_id": [], "score": []})

    X_common = X[common_mask]
    centroid = X_common.mean(axis=0)
    if not sparse.issparse(centroid):
        centroid = sparse.csr_matrix(centroid)
    else:
        centroid = centroid.tocsr()

    c_norm = float(np.sqrt(centroid.multiply(centroid).sum()))
    if c_norm <= 1e-12:
        return pd.DataFrame({"user_id": [], "score": []})

    cand_mask = ~common_mask
    cand_ids = all_user_ids[cand_mask]
    X_cand = X[cand_mask]

    row_norms = np.sqrt(X_cand.multiply(X_cand).sum(axis=1)).A1
    row_norms = np.maximum(row_norms, 1e-12)

    best_ids: list[int] = []
    best_scores: list[float] = []

    n = X_cand.shape[0]
    for start in tqdm(range(0, n, batch_size), desc="Scoring candidates", unit="batch"):
        end = min(start + batch_size, n)
        xb = X_cand[start:end]
        dots = xb @ centroid.T
        dots = np.asarray(dots.todense()).reshape(-1)
        sims = dots / (row_norms[start:end] * c_norm)

        ok = sims >= sim_threshold
        if not np.any(ok):
            continue

        ids_ok = cand_ids[start:end][ok]
        sims_ok = sims[ok]

        best_ids.extend(ids_ok.tolist())
        best_scores.extend(sims_ok.astype(np.float32).tolist())

    if not best_ids:
        return pd.DataFrame({"user_id": [], "score": []})

    ids = np.asarray(best_ids, dtype=np.int64)
    scores = np.asarray(best_scores, dtype=np.float32)

    if ids.shape[0] > top_k:
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        ids = ids[idx]
        scores = scores[idx]

    order = np.argsort(-scores, kind="mergesort")
    out = pd.DataFrame({"user_id": ids[order], "score": scores[order].astype(float)})
    return out


def centroid_candidates_hashtrick(
    df_users: pd.DataFrame,
    user_ids: Sequence[int] | np.ndarray,
    user_common: Sequence[int] | np.ndarray,
    top_k: int = 10_000,
    sim_threshold: float = 0.15,
    user_id_col: str = "user_id",
    num_cols: Sequence[str] | None = None,
    cat_cols: Sequence[str] | None = None,
    cfg: HashCentroidConfig = HashCentroidConfig(),
) -> pd.DataFrame:
    X, ordered_user_ids, used_num, used_cat = build_hashed_matrix(
        df=df_users,
        user_ids=user_ids,
        user_id_col=user_id_col,
        num_cols=num_cols,
        cat_cols=cat_cols,
        cfg=cfg,
    )
    res = topk_similar_to_centroid(
        X=X,
        all_user_ids=ordered_user_ids,
        user_common=user_common,
        top_k=int(top_k),
        sim_threshold=float(sim_threshold),
        batch_size=max(50_000, cfg.batch_size * 20),
    )
    return res