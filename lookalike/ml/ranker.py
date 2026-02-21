from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.linear_model import LogisticRegression


CAT_FEATURES = [
    "age_bucket",
    "gender_cd",
    "region",
    "segment",
    "region_size",
]


@dataclass
class RankerArtifacts:
    model: CatBoostRanker
    feature_names: list[str]
    calibrator: LogisticRegression


class RankerTrainer:
    def __init__(self, params):
        self.params = params

    def fit(self, train_df: pd.DataFrame, label_col: str = "label", group_col: str = "group_id") -> RankerArtifacts:
        features = [c for c in train_df.columns if c not in {label_col, group_col, "offer_id", "merchant_id_offer", "brand", "user_id"}]
        cats = [c for c in CAT_FEATURES if c in features]
        pool = Pool(
            data=train_df[features],
            label=train_df[label_col],
            group_id=train_df[group_col],
            cat_features=cats,
        )
        model = CatBoostRanker(
            loss_function=self.params.catboost_loss,
            iterations=self.params.catboost_iterations,
            depth=self.params.catboost_depth,
            learning_rate=self.params.catboost_learning_rate,
            l2_leaf_reg=self.params.catboost_l2_leaf_reg,
            random_seed=self.params.random_seed,
            verbose=False,
            task_type="GPU" if self._cuda_available() else "CPU",
        )
        model.fit(pool)
        calib = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000, random_state=self.params.random_seed)
        dummy_x = np.array([[0.0], [1.0]])
        dummy_y = np.array([0, 1])
        calib.fit(dummy_x, dummy_y)
        return RankerArtifacts(model=model, feature_names=features, calibrator=calib)

    def fit_calibrator(self, artifacts: RankerArtifacts, val_df: pd.DataFrame, y_val: pd.Series) -> None:
        raw = artifacts.model.predict(val_df[artifacts.feature_names])
        calib = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000, random_state=self.params.random_seed)
        calib.fit(raw.reshape(-1, 1), y_val.values)
        artifacts.calibrator = calib

    def predict_scores(self, artifacts: RankerArtifacts, df: pd.DataFrame) -> np.ndarray:
        raw = artifacts.model.predict(df[artifacts.feature_names])
        prob = artifacts.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return prob

    def reasons(self, artifacts: RankerArtifacts, top_df: pd.DataFrame, top_k: int) -> list[dict]:
        if top_df.empty:
            return []
        shap = artifacts.model.get_feature_importance(Pool(top_df[artifacts.feature_names]), type="ShapValues")
        shap = shap[:, :-1]
        mean_abs = np.mean(np.abs(shap), axis=0)
        mean_signed = np.mean(shap, axis=0)
        order = np.lexsort((np.array(artifacts.feature_names), -mean_abs))[:top_k]
        out = []
        for idx in order:
            name = artifacts.feature_names[idx]
            if name in CAT_FEATURES and name in top_df.columns:
                mode = top_df[name].astype(str).mode(dropna=False)
                val = mode.iloc[0] if len(mode) else "__MISSING__"
                feature_name = f"{name}={val}"
            else:
                feature_name = name
            out.append({"feature": feature_name, "impact": float(mean_signed[idx])})
        return out

    def _cuda_available(self) -> bool:
        try:
            from catboost.utils import get_gpu_device_count
            return get_gpu_device_count() > 0
        except Exception:
            return False
