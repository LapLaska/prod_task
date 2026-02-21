from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .text_embeddings import similar_offers


@dataclass
class CandidateResult:
    candidates: pd.DataFrame


class CandidateGenerator:
    def __init__(self, params, als_artifacts, text_artifacts, bundle, split_date=None):
        self.params = params
        self.als = als_artifacts
        self.text = text_artifacts
        self.bundle = bundle
        self.split_date = split_date
        self.user_to_idx = {u: i for i, u in enumerate(self.als.user_ids.tolist())}
        self.brand_to_idx = {b: i for i, b in enumerate(self.als.brand_ids.tolist())}
        self.popularity = self._build_popularity()
        self.max_pop_score = max(self.popularity["pop_score"].max(), 1.0) if len(self.popularity) > 0 else 1.0

    def _build_popularity(self) -> pd.DataFrame:
        tx = self.bundle.transaction.copy()
        ref_date = self.split_date if self.split_date is not None else tx["event_date"].max()
        lo = ref_date - pd.Timedelta(days=self.params.t_pop_days)
        tx = tx[(tx["event_date"] >= lo) & (tx["event_date"] < ref_date)]
        if tx.empty:
            return pd.DataFrame(columns=["user_id", "tx_count_30d", "nunique_brand_dk_30d", "pop_score", "rank_pop"])
        pop = tx.groupby("user_id").agg(
            tx_count_30d=("transaction_id", "count"),
            nunique_brand_dk_30d=("brand_dk", "nunique"),
        ).reset_index()
        pop["pop_score"] = pop["tx_count_30d"] * pop["nunique_brand_dk_30d"]
        pop = pop.sort_values(["pop_score", "tx_count_30d", "user_id"], ascending=[False, False, True]).reset_index(drop=True)
        pop["rank_pop"] = np.arange(1, len(pop) + 1)
        return pop

    def _offer_brand(self, offer_id: int, merchant_id: int) -> tuple[int, pd.Timestamp]:
        offer = self.bundle.offer[self.bundle.offer["offer_id"] == offer_id]
        if offer.empty:
            raise ValueError("offer_not_found")
        row = offer.iloc[0]
        if int(row["merchant_id_offer"]) != int(merchant_id):
            raise ValueError("merchant_offer_mismatch")
        merch = self.bundle.merchant[self.bundle.merchant["merchant_id_offer"] == merchant_id]
        if merch.empty:
            raise ValueError("merchant_not_found")
        brand = int(merch.iloc[0]["brand_dk"])
        start_date = pd.to_datetime(row["start_date"])
        return brand, start_date

    def _get_current_clients(self, brand: int, start_date: pd.Timestamp) -> set[int]:
        tx = self.bundle.transaction
        subset = tx[(tx["brand_dk"] == brand) & (tx["event_date"] < start_date)]
        return set(subset["user_id"].dropna().astype(int).tolist())

    def _brand_topk(self, brand: int) -> pd.DataFrame:
        if brand not in self.brand_to_idx:
            return pd.DataFrame(columns=["user_id", "score_brand", "rank_brand"])
        b_idx = self.brand_to_idx[brand]
        scores = self.als.user_factors @ self.als.brand_factors[b_idx]
        order = np.lexsort((self.als.user_ids, -scores))
        top = order[: self.params.topk_brand]
        out = pd.DataFrame({"user_id": self.als.user_ids[top], "score_brand": scores[top]})
        out["rank_brand"] = np.arange(1, len(out) + 1)
        return out

    def _seed_users(self, offer_id: int, brand: int, start_date: pd.Timestamp, similar_ids: list[int]) -> list[int]:
        seeds = []
        strength = {}
        for sid in similar_ids:
            offer_row = self.bundle.offer[self.bundle.offer["offer_id"] == sid]
            if offer_row.empty:
                continue
            s_start = offer_row.iloc[0]["start_date"]
            s_end = offer_row.iloc[0]["end_date"]
            act = self.bundle.offer_activation
            rew = self.bundle.offer_reward
            s1 = set(act[(act["offer_id"] == sid) & (act["activation_date"] >= s_start) & (act["activation_date"] <= s_end)]["user_id"].dropna().astype(int))
            s2 = set(rew[(rew["offer_id"] == sid) & (rew["event_date"] >= s_start) & (rew["event_date"] <= s_end)]["user_id"].dropna().astype(int))
            tx = self.bundle.transaction
            pos = tx[(tx["brand_dk"] == brand) & (tx["event_date"] >= s_start) & (tx["event_date"] <= s_end)]["user_id"].dropna().astype(int)
            s3 = set(pos.tolist())
            group = s1.union(s2).union(s3)
            for u in group:
                strength[u] = strength.get(u, 0) + 1
            seeds.extend(group)
        seed_unique = sorted(set(seeds))
        seed_unique = [u for u in seed_unique if u in self.user_to_idx]
        if len(seed_unique) > 20000:
            seed_unique = sorted(seed_unique, key=lambda x: (-strength.get(x, 0), x))[:20000]
        return seed_unique

    def _seed_topk(self, offer_id: int, brand: int, start_date: pd.Timestamp) -> pd.DataFrame:
        sims = similar_offers(self.text, offer_id, self.params.similar_offers_k)
        seeds = self._seed_users(offer_id, brand, start_date, sims)
        if len(seeds) == 0:
            return pd.DataFrame(columns=["user_id", "score_seed", "rank_seed"])
        idx = np.array([self.user_to_idx[u] for u in seeds])
        centroid = self.als.user_factors[idx].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        scores = self.als.user_factors @ centroid
        order = np.lexsort((self.als.user_ids, -scores))
        top = order[: self.params.topk_seed]
        out = pd.DataFrame({"user_id": self.als.user_ids[top], "score_seed": scores[top]})
        out["rank_seed"] = np.arange(1, len(out) + 1)
        return out

    def generate(self, merchant_id: int, offer_id: int) -> CandidateResult:
        brand, start_date = self._offer_brand(offer_id, merchant_id)
        brand_df = self._brand_topk(brand)
        seed_df = self._seed_topk(offer_id, brand, start_date)
        merged = pd.merge(brand_df, seed_df, on="user_id", how="outer")
        merged["rank_brand"] = merged["rank_brand"].fillna(1e9)
        merged["rank_seed"] = merged["rank_seed"].fillna(1e9)
        merged["score_brand"] = merged["score_brand"].fillna(0.0)
        merged["score_seed"] = merged["score_seed"].fillna(0.0)
        if len(merged) < self.params.target_candidates:
            merged = merged.merge(self.popularity[["user_id", "pop_score", "rank_pop"]], on="user_id", how="outer")
        else:
            merged = merged.merge(self.popularity[["user_id", "pop_score", "rank_pop"]], on="user_id", how="left")
        merged["rank_pop"] = merged["rank_pop"].fillna(1e9)
        merged["pop_score"] = merged["pop_score"].fillna(0.0)
        merged["retrieval_rank"] = merged[["rank_brand", "rank_seed", "rank_pop"]].min(axis=1)
        current_clients = self._get_current_clients(brand, start_date)
        merged = merged[~merged["user_id"].astype(int).isin(current_clients)].copy()
        merged["pop_score_norm"] = merged["pop_score"] / self.max_pop_score
        merged["retrieval_score"] = merged[["score_brand", "score_seed", "pop_score_norm"]].max(axis=1)
        merged = merged.sort_values(["retrieval_rank", "retrieval_score", "user_id"], ascending=[True, False, True])
        merged = merged.head(self.params.max_candidates).reset_index(drop=True)
        merged["brand"] = brand
        merged["offer_id"] = offer_id
        merged["merchant_id_offer"] = merchant_id
        return CandidateResult(candidates=merged)
