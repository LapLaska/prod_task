from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np


DATE_COLS = {
    "people": ["last_activity_day"],
    "transaction": ["event_date"],
    "offer": ["start_date", "end_date"],
    "offer_seens": ["start_date", "end_date"],
    "offer_activation": ["activation_date"],
    "offer_reward": ["event_date"],
    "receipts": ["date_operated"],
}


PANDAS_MAX_TS = pd.Timestamp.max.normalize()


@dataclass
class DataBundle:
    people: pd.DataFrame
    segments: pd.DataFrame
    transaction: pd.DataFrame
    offer: pd.DataFrame
    merchant: pd.DataFrame
    financial_account: pd.DataFrame
    offer_seens: pd.DataFrame
    offer_activation: pd.DataFrame
    offer_reward: pd.DataFrame
    receipts: pd.DataFrame


def parse_dates(bundle: DataBundle) -> DataBundle:
    for table_name, cols in DATE_COLS.items():
        df = getattr(bundle, table_name)
        for col in cols:
            if col in df.columns:
                if table_name == "offer" and col == "end_date":
                    original = df[col]
                    raw = original.astype("string").str.strip()
                    is_infinite = original.isna() | (raw == "") | (raw == "5999-01-01")
                    parsed = pd.to_datetime(original, errors="coerce")
                    parsed.loc[is_infinite] = PANDAS_MAX_TS
                    df[col] = parsed
                else:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
    return bundle


def add_brand_conflict_features(merchant: pd.DataFrame) -> pd.DataFrame:
    counts = merchant.groupby("brand_dk")["merchant_id_offer"].nunique().rename("n_merchants_for_brand")
    out = merchant.merge(counts, on="brand_dk", how="left")
    out["brand_conflict_flg"] = (out["n_merchants_for_brand"] > 1).astype(int)
    return out


def split_offers(offer: pd.DataFrame) -> tuple[list[int], list[int], pd.Timestamp]:
    ordered = offer[["offer_id", "start_date"]].dropna(subset=["offer_id", "start_date"]).copy()
    ordered = ordered.sort_values(["start_date", "offer_id"]).reset_index(drop=True)
    offer_ids = ordered["offer_id"].astype(int).tolist()
    split_idx = int(np.floor(0.8 * len(offer_ids)))
    split_idx = max(1, min(split_idx, len(offer_ids) - 1))
    train_offers = offer_ids[:split_idx]
    val_offers = offer_ids[split_idx:]
    split_date = ordered.loc[split_idx, "start_date"]
    return train_offers, val_offers, split_date


def apply_nan_policy(df: pd.DataFrame, nan_policy: str) -> pd.DataFrame:
    out = df.copy()
    if nan_policy == "keep":
        return out
    for col in out.columns:
        if out[col].isna().any():
            out[f"is_null_{col}"] = out[col].isna().astype(int)
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(0)
            else:
                out[col] = out[col].fillna("__MISSING__")
    return out
