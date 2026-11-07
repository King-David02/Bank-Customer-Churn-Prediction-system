from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

def build_xgb_classifier(params: Dict[str, Any] | None = None, seed: int =42)-> Any:
    if not XGBOOST_AVAILABLE:
        raise ImportError("failed to import xgboost try 'uv add xgboost' or 'pip install xgboost' ")
    
    defaults = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": -1,
    }
    
    if params:
        defaults.update(params)
    
    return xgb.XGBClassifier(**defaults)

def build_lgb_classifier(params: Dict[str, Any] | None = None, seed: int = 42) -> Any:
    if not LGB_AVAILABLE:
        raise ImportError("failed to import lightgbm try 'uv add lightgmb' or 'pip install lightgbm' ")
    
    defaults = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "random_state": seed,
    "n_jobs": -1,
    "verbose": -1,
}
    if params:
        defaults.update(params)
    
    return lgb.LGBMClassifier(**defaults)