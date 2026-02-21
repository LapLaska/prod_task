from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .config import load_params
from .utils import ensure_dir, set_global_seed
from .data_prep import DataBundle, parse_dates, split_offers, add_brand_conflict_features, apply_nan_policy
from .als_model import AlsTrainer
from .text_embeddings import OfferTextEncoder
from .retrieval import CandidateGenerator
from .features import FeatureBuilder, build_labels
from .ranker import RankerTrainer


@dataclass
class PipelineArtifacts:
    params: object
    als: object
    text: object
    ranker: object
    split_date: pd.Timestamp
    train_offers: list[int]
    val_offers: list[int]


class LookalikePipeline:
    def __init__(self, params_path: str = "params.yaml", artifacts_dir: str = "artifacts/lookalike_v2"):
        self.params = load_params(params_path)
        self.artifacts_dir = ensure_dir(artifacts_dir)
        set_global_seed(self.params.random_seed)
        self.artifacts = None

    def train(self, bundle: DataBundle) -> dict:
        bundle = parse_dates(bundle)
        bundle.merchant = add_brand_conflict_features(bundle.merchant)
        train_offers, val_offers, split_date = split_offers(bundle.offer)
        tx_train = bundle.transaction[bundle.transaction["event_date"] < split_date].copy()
        rc_train = bundle.receipts[bundle.receipts["date_operated"] < split_date].copy()
        snapshot = DataBundle(
            people=bundle.people,
            segments=bundle.segments,
            transaction=tx_train,
            offer=bundle.offer,
            merchant=bundle.merchant,
            financial_account=bundle.financial_account,
            offer_seens=bundle.offer_seens,
            offer_activation=bundle.offer_activation[bundle.offer_activation["activation_date"] < split_date],
            offer_reward=bundle.offer_reward[bundle.offer_reward["event_date"] < split_date],
            receipts=rc_train,
        )
        als = AlsTrainer(self.params).fit(tx_train)
        text = OfferTextEncoder(self.params).fit(bundle.offer[bundle.offer["offer_id"].isin(train_offers)].copy())
        gen = CandidateGenerator(self.params, als, text, snapshot, split_date=split_date)
        feat = FeatureBuilder(snapshot, als)
        train_groups = []
        for offer_id in tqdm(train_offers, desc="Build train groups"):
            offer_row = snapshot.offer[snapshot.offer["offer_id"] == offer_id]
            if offer_row.empty:
                continue
            merchant_id = int(offer_row.iloc[0]["merchant_id_offer"])
            candidates = gen.generate(merchant_id=merchant_id, offer_id=int(offer_id)).candidates
            x = feat.build(candidates, offer_row.iloc[0])
            y = build_labels(x, snapshot.offer, snapshot.transaction)
            x["label"] = y
            positives = x[x["label"] == 1]
            if positives.empty:
                continue
            negatives = x[x["label"] == 0].copy()
            negatives = negatives.sort_values(["retrieval_rank", "retrieval_score", "user_id"], ascending=[True, False, True])
            negatives = negatives.head(min(20 * len(positives), len(negatives)))
            group = pd.concat([positives, negatives], axis=0).copy()
            group["group_id"] = int(offer_id)
            train_groups.append(group)
        train_df = pd.concat(train_groups, axis=0).reset_index(drop=True)
        train_df = apply_nan_policy(train_df, self.params.nan_policy)
        ranker = RankerTrainer(self.params)
        ranker_art = ranker.fit(train_df)
        val_groups = []
        for offer_id in tqdm(val_offers, desc="Build val groups"):
            offer_row = snapshot.offer[snapshot.offer["offer_id"] == offer_id]
            if offer_row.empty:
                continue
            merchant_id = int(offer_row.iloc[0]["merchant_id_offer"])
            candidates = gen.generate(merchant_id=merchant_id, offer_id=int(offer_id)).candidates
            x = feat.build(candidates, offer_row.iloc[0])
            y = build_labels(x, bundle.offer, bundle.transaction)
            x["label"] = y
            x["group_id"] = int(offer_id)
            val_groups.append(x)
        val_df = pd.concat(val_groups, axis=0).reset_index(drop=True)
        val_df = apply_nan_policy(val_df, self.params.nan_policy)
        ranker.fit_calibrator(ranker_art, val_df, val_df["label"])
        map100 = self._mapk(ranker, ranker_art, val_df, k=100)
        self.artifacts = PipelineArtifacts(
            params=self.params,
            als=als,
            text=text,
            ranker=ranker_art,
            split_date=split_date,
            train_offers=train_offers,
            val_offers=val_offers,
        )
        self._save_meta(map100)
        return {"map_at_100": map100, "split_date": str(split_date.date())}

    def predict(self, bundle: DataBundle, merchant_id: int, offer_id: int, top_n: int) -> dict:
        if self.artifacts is None:
            raise RuntimeError("model_not_trained")
        if top_n < 1 or top_n > 1000:
            raise ValueError("top_n_out_of_range")
        bundle = parse_dates(bundle)
        bundle.merchant = add_brand_conflict_features(bundle.merchant)
        gen = CandidateGenerator(self.params, self.artifacts.als, self.artifacts.text, bundle, split_date=None)
        offer_row = bundle.offer[bundle.offer["offer_id"] == offer_id]
        if offer_row.empty:
            raise ValueError("offer_not_found")
        feat = FeatureBuilder(bundle, self.artifacts.als)
        cand = gen.generate(merchant_id=merchant_id, offer_id=offer_id).candidates
        x = feat.build(cand, offer_row.iloc[0])
        x = apply_nan_policy(x, self.params.nan_policy)
        ranker = RankerTrainer(self.params)
        x["score"] = ranker.predict_scores(self.artifacts.ranker, x)
        x = x.sort_values(["score", "user_id"], ascending=[False, True]).head(top_n)
        reasons = ranker.reasons(self.artifacts.ranker, x, self.params.reasons_top_k)
        audience = [{"user_id": int(u), "score": float(s)} for u, s in zip(x["user_id"], x["score"])]
        return {"offer_id": offer_id, "merchant_id": merchant_id, "audience": audience, "audience_size": len(audience), "reasons": reasons}

    def _mapk(self, ranker, artifacts, val_df: pd.DataFrame, k: int = 100) -> float:
        scored = val_df.copy()
        scored["score"] = ranker.predict_scores(artifacts, scored)
        aps = []
        for gid, gdf in scored.groupby("group_id"):
            gdf = gdf.sort_values(["score", "user_id"], ascending=[False, True]).head(k)
            hits = 0
            precisions = []
            for i, y in enumerate(gdf["label"].tolist(), start=1):
                if y == 1:
                    hits += 1
                    precisions.append(hits / i)
            denom = max(1, min(int((val_df[val_df["group_id"] == gid]["label"] == 1).sum()), k))
            aps.append(float(np.sum(precisions) / denom if precisions else 0.0))
        return float(np.mean(aps) if aps else 0.0)

    def _save_meta(self, map100: float) -> None:
        payload = {"map_at_100": map100, "params": self.params.__dict__}
        out = Path(self.artifacts_dir) / "metrics.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
