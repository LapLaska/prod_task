from __future__ import annotations
import numpy as np
import pandas as pd


WINDOWS = [15, 30, 60, 90]


class FeatureBuilder:
    def __init__(self, bundle, als_artifacts):
        self.bundle = bundle
        self.als = als_artifacts
        self.user_to_idx = {u: i for i, u in enumerate(self.als.user_ids.tolist())}
        self.brand_to_idx = {b: i for i, b in enumerate(self.als.brand_ids.tolist())}

    def _window_mask(self, s: pd.Series, ref_date: pd.Timestamp, w: int) -> pd.Series:
        return (s >= ref_date - pd.Timedelta(days=w)) & (s < ref_date)

    def _mode_share(self, s: pd.Series):
        if len(s) == 0:
            return np.nan, np.nan
        vc = s.value_counts(dropna=False)
        return vc.index[0], vc.iloc[0] / len(s)

    def build(self, candidates: pd.DataFrame, offer_row: pd.Series) -> pd.DataFrame:
        ref_date = offer_row["start_date"]
        out = candidates.copy()
        out = out.merge(self.bundle.people, on="user_id", how="left")
        out = out.merge(self.bundle.segments, on="user_id", how="left")
        fa = self.bundle.financial_account.groupby("user_id").agg(
            fa_n_accounts_total=("product_cd", "size"),
            fa_nunique_product_cd=("product_cd", "nunique"),
        ).reset_index()
        out = out.merge(fa, on="user_id", how="left")
        tx = self.bundle.transaction[self.bundle.transaction["event_date"] < ref_date].copy()
        rc = self.bundle.receipts[self.bundle.receipts["date_operated"] < ref_date].copy()
        full_tx = tx.groupby("user_id").agg(
            transaction_count=("transaction_id", "count"),
            merchant_tx_count=("merchant_id_tx", "nunique"),
            nunique_brand_dk=("brand_dk", "nunique"),
        ).reset_index()
        out = out.merge(full_tx, on="user_id", how="left")
        for w in WINDOWS:
            txw = tx[self._window_mask(tx["event_date"], ref_date, w)]
            agg = txw.groupby("user_id").agg(
                **{
                    f"transaction_count_{w}d": ("transaction_id", "count"),
                    f"merchant_tx_count_{w}d": ("merchant_id_tx", "nunique"),
                    f"nunique_brand_dk_{w}d": ("brand_dk", "nunique"),
                }
            ).reset_index()
            out = out.merge(agg, on="user_id", how="left")
            rcw = rc[self._window_mask(rc["date_operated"], ref_date, w)]
            racc = rcw.groupby("user_id").agg(
                **{
                    f"receipt_count_{w}d": ("category_name", "count"),
                    f"active_days_receipt_{w}d": ("date_operated", "nunique"),
                }
            ).reset_index()
            out = out.merge(racc, on="user_id", how="left")
            out[f"log_receipt_count_{w}d"] = np.log1p(out[f"receipt_count_{w}d"].fillna(0))
        out["offer_duration_days"] = (offer_row["end_date"] - offer_row["start_date"]).days
        text = offer_row.get("offer_text", "")
        if pd.isna(text):
            text = ""
        out["offer_text_len"] = len(str(text))
        out["offer_text_missing"] = int(len(str(text)) == 0)
        out["days_since_last_activity"] = (ref_date - out["last_activity_day"]).dt.days
        out["dot_user_brand"] = out.apply(self._dot_user_brand, axis=1)
        out["dot_user_centroid"] = out.get("score_seed", 0.0)
        return out

    def _dot_user_brand(self, row: pd.Series) -> float:
        user_id = int(row["user_id"])
        brand = int(row["brand"])
        if user_id not in self.user_to_idx or brand not in self.brand_to_idx:
            return 0.0
        u = self.als.user_factors[self.user_to_idx[user_id]]
        b = self.als.brand_factors[self.brand_to_idx[brand]]
        return float(np.dot(u, b))


def build_labels(df: pd.DataFrame, offer_df: pd.DataFrame, transaction: pd.DataFrame) -> pd.Series:
    lookup_offer = offer_df.set_index("offer_id")
    tx = transaction.copy()
    labels = []
    for _, row in df.iterrows():
        offer = lookup_offer.loc[int(row["offer_id"])]
        brand = int(row["brand"])
        user = int(row["user_id"])
        start = offer["start_date"]
        end = offer["end_date"]
        user_tx = tx[(tx["user_id"] == user) & (tx["brand_dk"] == brand)]
        in_window = ((user_tx["event_date"] >= start) & (user_tx["event_date"] <= end)).any()
        before = (user_tx["event_date"] < start).any()
        labels.append(int(in_window and not before))
    return pd.Series(labels, index=df.index)
