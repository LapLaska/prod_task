from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from tqdm.auto import tqdm


@dataclass
class AlsArtifacts:
    user_ids: np.ndarray
    brand_ids: np.ndarray
    user_factors: np.ndarray
    brand_factors: np.ndarray


class AlsTrainer:
    def __init__(self, params):
        self.params = params

    def fit(self, transaction: pd.DataFrame) -> AlsArtifacts:
        tx = transaction[["user_id", "brand_dk", "event_date"]].dropna().copy()
        tx["event_date"] = pd.to_datetime(tx["event_date"], errors="coerce")
        tx = tx.dropna(subset=["event_date"])
        user_ids = np.array(sorted(tx["user_id"].unique()))
        brand_ids = np.array(sorted(tx["brand_dk"].unique()))
        user_to_idx = {u: i for i, u in enumerate(user_ids)}
        brand_to_idx = {b: i for i, b in enumerate(brand_ids)}
        tx["u_idx"] = tx["user_id"].map(user_to_idx)
        tx["b_idx"] = tx["brand_dk"].map(brand_to_idx)
        if self.params.als_mode == "binary":
            pairs = tx[["u_idx", "b_idx"]].drop_duplicates()
            values = np.ones(len(pairs), dtype=np.float32)
            mat = coo_matrix((values, (pairs["u_idx"], pairs["b_idx"])), shape=(len(user_ids), len(brand_ids))).tocsr()
        else:
            ref_date = tx["event_date"].max()
            days = (ref_date - tx["event_date"]).dt.days.clip(lower=0)
            tx["w"] = np.exp(-days / self.params.als_time_decay_tau_days)
            grouped = tx.groupby(["u_idx", "b_idx"]).agg(cnt=("event_date", "size"), wsum=("w", "sum")).reset_index()
            grouped["val"] = np.log1p(grouped["cnt"].astype(float)) * grouped["wsum"]
            mat = coo_matrix(
                (grouped["val"].astype(np.float32), (grouped["u_idx"], grouped["b_idx"])),
                shape=(len(user_ids), len(brand_ids)),
            ).tocsr()
        conf: csr_matrix = mat * self.params.als_alpha
        model = AlternatingLeastSquares(
            factors=self.params.als_factors,
            iterations=self.params.als_iterations,
            regularization=self.params.als_regularization,
            random_state=self.params.random_seed,
        )
        for _ in tqdm(range(1), desc="ALS fit"):
            model.fit(conf)
        user_factors = model.user_factors.astype(np.float32)
        brand_factors = model.item_factors.astype(np.float32)
        user_factors = user_factors / np.maximum(np.linalg.norm(user_factors, axis=1, keepdims=True), 1e-12)
        brand_factors = brand_factors / np.maximum(np.linalg.norm(brand_factors, axis=1, keepdims=True), 1e-12)
        return AlsArtifacts(user_ids=user_ids, brand_ids=brand_ids, user_factors=user_factors, brand_factors=brand_factors)
