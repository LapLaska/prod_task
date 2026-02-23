from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import numpy as np


@dataclass(slots=True)
class FeatureMakerConfig:
    threshold_date: str | pd.Timestamp


class FeatureMaker:
    def __init__(self, threshold_date: str | pd.Timestamp):
        self.threshold_date = pd.to_datetime(threshold_date)

    def get_cat_features(
        self,
        df: pd.DataFrame,
        date: str,
        cats: str | list[str],
        df_name: str,
        cat_names: str | list[str] | None = None,
        train_test_split: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()
        df[date] = pd.to_datetime(df[date])

        if train_test_split:
            df = df[df[date] < self.threshold_date]

        d = df.groupby("user_id")[date].agg(min_date="min", max_date="max", active_days="nunique")
        d[f"span_days_{df_name}"] = (d["max_date"] - d["min_date"]).dt.days + 1
        d[f"days_ratio_{df_name}"] = d["active_days"] / d[f"span_days_{df_name}"]
        d = d.rename(
            columns={
                "min_date": f"min_date_{df_name}",
                "max_date": f"max_date_{df_name}",
                "active_days": f"active_days_{df_name}",
            }
        )

        if isinstance(cats, str):
            cats = [cats]

        if cat_names is None:
            cat_names = list(cats)
        else:
            if isinstance(cat_names, str):
                cat_names = [cat_names]
            if len(cat_names) != len(cats):
                raise ValueError("len(cat_names) must equal len(cats)")
            if len(set(cat_names)) != len(cat_names):
                raise ValueError("cat_names must be unique")

        out = d

        totals = df.groupby("user_id").size().rename("total")

        for cat, cat_alias in zip(cats, cat_names):
            counts = df.groupby(["user_id", cat]).size().rename("cnt").reset_index()
            idx = counts.groupby("user_id")["cnt"].idxmax()
            top = counts.loc[idx, ["user_id", cat, "cnt"]].set_index("user_id")

            out = out.join(
                top.rename(columns={cat: f"top_{cat_alias}_{df_name}"}).drop(columns="cnt"),
                how="left",
            )

            out = out.join(
                (top["cnt"] / totals).rename(f"top_ratio_{cat_alias}_{df_name}"),
                how="left",
            )

        return out.reset_index()

    def get_trans_features(
        self,
        transaction_v1: pd.DataFrame,
        name: str,
        threshold: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        transaction_v1['event_date'] = pd.to_datetime(transaction_v1['event_date'])
    
        if threshold is None:
            t = transaction_v1[transaction_v1["event_date"] < self.threshold_date].copy()
        else:
            thr = pd.to_datetime(threshold)
            t = transaction_v1[transaction_v1["event_date"].between(thr, self.threshold_date)].copy()

        dff = pd.DataFrame()
        dff[f"mostpop_amount_bucket_{name}"] = t.groupby("user_id")["amount_bucket"].agg(lambda x: x.mode().iloc[0])
        dff[f"online_transaction_rate_{name}"] = t.groupby("user_id")["online_transaction_flg"].agg(
            lambda x: (x == "Y").mean()
        )
        dff[f"nunique_brand_dk_{name}"] = t.groupby("user_id")["brand_dk"].nunique()
        dff[f"transaction_count_{name}"] = t.groupby("user_id")["transaction_id"].nunique()
        dff[f"merchant_tx_count_{name}"] = t.groupby("user_id")["merchant_id_tx"].nunique()

        result = t.groupby("user_id")["event_date"].agg(
            **{
                f"min_date_trans_{name}": "min",
                f"max_date_trans_{name}": "max",
            }
        )
        result[f"span_days_trans_{name}"] = (result[f"max_date_trans_{name}"] - result[f"min_date_trans_{name}"]).dt.days + 1

        dff = dff.merge(result, on="user_id")
        return dff
    
    def get_count_log1p(
        self,
        df: pd.DataFrame
    ):
        df = df.copy()
        cc = [col for col in df.columns if 'count' in col and not 'log1p' in col]
        for col in cc:
            if f'{col}_log1p' not in df.columns:
                df[f'{col}_log1p'] = np.log1p(df[col])
        return df
    def clean(
        self,
        df: pd.DataFrame
    ):
        df = df.copy()
        dt_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        cols = [col for col in df.columns if col not in dt_cols]
        return df[cols]
    
    
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_dir", type=str, default='data/dir')
#     ap.add_argument("--save_dir", type=str, required='work/features')
#     ap.add_argument("--trans", type=str|list[str], default='transactions.csv')
#     ap.add_argument("--cats", type=str|list[str], default='transactions.csv')
#     args = ap.parse_args()

#     p = read_params(args.params)