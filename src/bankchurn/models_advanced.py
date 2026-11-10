from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from typing import Any
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
    
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def build_xgb_classifier(params: dict[str, Any] | None = None, seed: int =42)-> Any:
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

def build_lgb_classifier(params: dict[str, Any] | None = None, seed: int = 42) -> Any:
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

def build_mlp_classifier(params: dict[str, Any] | None = None, seed: int = 42) -> Any:
    
    defaults = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "max_iter": 500,
    "early_stopping": True,
    "n_iter_no_change": 20,
    "random_state": seed,
}
    if params:
        defaults.update(params)
    
    return MLPClassifier(**defaults)

def build_lr(params: dict[str, Any] | None = None, seed: int = 42) -> Any:
    defaults = {"max_iter": 1000, "random_state": seed, "class_weight": "balanced"}
    
    if params:
        defaults.update(params)
    return LogisticRegression(**defaults)

def build_rf(params: dict[str, Any] | None = None, seed: int = 42) -> Any:
    
    defaults = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": seed,
    "n_jobs": -1,
}
    if params:
        defaults.update(params)
    return RandomForestClassifier(**defaults)

def build_gb_classifier(params: dict[str, Any] | None = None, seed: int = 42):
    
    defaults = {
        "loss": "log_loss",
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0,
        "max_depth": 3,
        "min_impurity_decrease": 0,
    }
    if params:
        defaults.update(params)
    return GradientBoostingClassifier(**defaults)

def build_ensemble_classifier(params: dict[str, Any] |None = None, seed: int = 42):
    params = params or {}
    weight = params.get("weights", [0.4, 0.6])
    voting = params.get("voting", "soft")
    lr = build_lr(params=params.get("lr_params"), seed=seed)
    rf =build_rf(params=params.get("rf_params"), seed=seed)
    return VotingClassifier(estimators=[("lr", lr), ("rf", rf)], weights=weight, voting=voting)

def build_stacking_classifier(params: dict[str, Any] | None = None, seed: int = 42):
    params = params or {}
    base_estimators = []

    base_estimators.append(("lr", build_lr(params=params.get("lr_params"), seed = seed)))
    base_estimators.append(("rf", build_rf(params=params.get("rf_params"), seed = seed)))
    base_estimators.append(("gb", build_gb_classifier(params=params.get("gb_params"), seed = seed)))
    if XGBOOST_AVAILABLE:
        base_estimators.append(("xgb", build_xgb_classifier(params=params.get("xgb_params"), seed = seed)))
    if LGB_AVAILABLE:
        base_estimators.append(("lgb", build_lgb_classifier(params=params.get("lgb_params"), seed = seed)))
    
    meta_learner = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=seed
    )
    
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )
    
    use_calibration = params.get("calibrate", True)
    if use_calibration:
        return CalibratedClassifierCV(
            stacker,
            cv=3,
            method="isotonic",
        )
    return stacker
    
if TORCH_AVAILABLE:

    class _ChurnNet(nn.Module):
        """Feed-forward neural network for tabular churn prediction.

        Architecture
        ------------
        Input → BatchNorm → Linear(d,128) → ReLU → Dropout(0.3)
              → Linear(128,64) → ReLU → Dropout(0.2)
              → Linear(64,32)  → ReLU → Dropout(0.1)
              → Linear(32,2)   → output (logits)
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int] | None = None,
            dropout: float = 0.3,
        ):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [128, 64, 32]

            layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
            prev_dim = input_dim
            for i, h_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                drop_rate = max(dropout - i * 0.1, 0.05)
                layers.append(nn.Dropout(drop_rate))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, 2))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

    class TorchTabularClassifier(BaseEstimator, ClassifierMixin):
        """Sklearn-compatible wrapper for PyTorch tabular classifier.

        Implements fit/predict/predict_proba so it can be used inside
        sklearn Pipelines just like any other estimator.

        Parameters
        ----------
        hidden_dims : list[int]
            Hidden layer dimensions.
        lr : float
            Learning rate.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        dropout : float
            Base dropout rate.
        weight_decay : float
            L2 regularization.
        early_stopping_patience : int
            Stop training if val loss does not improve for this many epochs.
        class_weight : str or None
            If 'balanced', compute class weights from training data.
        random_state : int
            Random seed.
        verbose : bool
            Print training progress.
        """

        def __init__(
            self,
            hidden_dims: list[int] | None = None,
            lr: float = 1e-3,
            epochs: int = 100,
            batch_size: int = 256,
            dropout: float = 0.3,
            weight_decay: float = 1e-4,
            early_stopping_patience: int = 10,
            class_weight: str | None = "balanced",
            random_state: int = 42,
            verbose: bool = False,
        ):
            self.hidden_dims = hidden_dims or [128, 64, 32]
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.dropout = dropout
            self.weight_decay = weight_decay
            self.early_stopping_patience = early_stopping_patience
            self.class_weight = class_weight
            self.random_state = random_state
            self.verbose = verbose
            self.classes_ = np.array([0, 1])
            self.model_: _ChurnNet | None = None
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchTabularClassifier":
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            X_t = self._to_tensor(X, dtype=torch.float32)
            y_t = self._to_tensor(y, dtype=torch.long)

            self.classes_ = np.unique(y)
            input_dim = X_t.shape[1]
            self.model_ = _ChurnNet(input_dim, self.hidden_dims, self.dropout).to(self.device_)

            # Class weights for imbalanced data
            if self.class_weight == "balanced":
                class_counts = np.bincount(y.astype(int))
                weights = torch.tensor(len(y) / (len(self.classes_) * class_counts), dtype=torch.float32).to(
                    self.device_
                )
            else:
                weights = None

            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

            # Train/val split for early stopping (use 10% of training data)
            n = len(X_t)
            val_size = max(int(n * 0.1), 1)
            indices = torch.randperm(n)
            train_idx, val_idx = indices[val_size:], indices[:val_size]

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            dataset = torch.utils.data.TensorDataset(X_t[train_idx], y_t[train_idx])
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model_.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in loader:
                    batch_X, batch_y = batch_X.to(self.device_), batch_y.to(self.device_)
                    optimizer.zero_grad()
                    logits = self.model_(batch_X)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                # Validation
                self.model_.eval()
                with torch.no_grad():
                    val_X = X_t[val_idx].to(self.device_)
                    val_y = y_t[val_idx].to(self.device_)
                    val_logits = self.model_(val_X)
                    val_loss = criterion(val_logits, val_y).item()
                self.model_.train()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    patience_counter += 1

                if self.verbose and epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{self.epochs} - loss: {epoch_loss/len(loader):.4f} - "
                        f"val_loss: {val_loss:.4f}"
                    )

                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break

            if best_state is not None:
                self.model_.load_state_dict(best_state)

            self.model_.eval()
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            if self.model_ is None:
                raise RuntimeError("Model not fitted. Call fit() first.")

            self.model_.eval()
            X_t = self._to_tensor(X, dtype=torch.float32).to(self.device_)

            with torch.no_grad():
                logits = self.model_(X_t)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
            return proba

        def predict(self, X: np.ndarray) -> np.ndarray:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        def _to_tensor(self, data: Any, dtype: torch.dtype) -> torch.Tensor:
            if isinstance(data, torch.Tensor):
                return data.to(dtype)
            arr = np.asarray(data, dtype=np.float32) if dtype == torch.float32 else np.asarray(data)
            return torch.tensor(arr, dtype=dtype)
        
AVAILABLE_MODELS = {
    "logistic_regression": "sklearn",
    "random_forest": "sklearn",
    "ensemble": "sklearn",
    "stacking": "sklearn",
    "mlp": "sklearn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "neural_network": "torch",
    "gradient_boosting": "lightgbm"
    
}

def build_model(
    model_type: str,
    params: dict[str, Any] | None = None,
    seed: int = 42
) -> BaseEstimator:
    params = params or {}
    
    if model_type == "xgboost":
        return build_xgb_classifier(params=params, seed=seed)
    elif model_type == "lightgbm":
        return build_lgb_classifier(params=params,seed = seed)
    elif model_type == "mlp":
        return build_mlp_classifier(params=params, seed = seed)
    elif model_type =="logistic_regression":
        return build_lr(params=params, seed = seed)
    elif model_type == "random_forest":
        return build_rf(params=params, seed = seed)
    elif model_type == "gradient_boosting":
        return build_gb_classifier(params=params, seed = seed)
    elif model_type == "ensemble":
        return build_ensemble_classifier(params=params, seed = seed)
    elif model_type == "stacking":
        return build_stacking_classifier(params=params, seed = seed)
    elif model_type == "neural_network":
        if not TORCH_AVAILABLE:
            raise ImportError("torch is not installed. Install with: pip install torch")
        return TorchTabularClassifier(random_state=seed, **params)

    
    else:
        raise ValueError(f"Unknown model: '{model_type}'. Available: {list(AVAILABLE_MODELS.keys())}")


def get_available_models() -> dict[str, bool]:
    """Return availability status of each model type."""
    return {
        "logistic_regression": True,
        "random_forest": True,
        "ensemble": True,
        "stacking": True,
        "gradient_boosting": True,
        "mlp": True,
        "xgboost": XGBOOST_AVAILABLE,
        "lightgbm": LGB_AVAILABLE,
        "neural_network": TORCH_AVAILABLE,
    }