from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Params:
    random_seed: int = 42
    offer_text_max_length: int = 256
    similar_offers_k: int = 50
    als_mode: str = "weighted"
    als_factors: int = 128
    als_iterations: int = 50
    als_regularization: float = 0.01
    als_alpha: float = 40.0
    als_time_decay_tau_days: float = 90.0
    topk_brand: int = 4000
    topk_seed: int = 4000
    target_candidates: int = 10000
    max_candidates: int = 10000
    t_pop_days: int = 30
    catboost_loss: str = "YetiRank"
    catboost_iterations: int = 500
    catboost_depth: int = 8
    catboost_learning_rate: float = 0.05
    catboost_l2_leaf_reg: float = 3.0
    calibration_method: str = "platt_logreg_1d"
    reasons_top_k: int = 3
    nan_policy: str = "impute"
    text_model_name: str = "deepvk/USER-base"
    text_batch_size: int = 64


def load_params(path: str | Path = "params.yaml") -> Params:
    file_path = Path(path)
    if not file_path.exists():
        return Params()
    raw = yaml.safe_load(file_path.read_text()) or {}
    return Params(**raw)
