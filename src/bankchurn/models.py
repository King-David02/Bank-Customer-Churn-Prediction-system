from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

class ResamplingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        strategy: str = "none",
        random_state: int = 42
    ):
        self.estimator = estimator
        self.strategy = strategy
        self.random_state = random_state
        
    
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> ResamplingClassifier:
        from sklearn.linear_model import LogisticRegression
        if self.estimator is None:
            self.estimator = LogisticRegression(random_state=self.random_state)
        else:    
            self.estimator_= self.estimator
        
        resampled_X, resampled_y = self._apply_resampling(X, y)
        self.estimator_.fit(resampled_X, resampled_y)
        
        logger.info(
            f"ResampleClassifier fitted with strategy='{self.strategy}' "
            f"(original samples: {len(y)}, resampled: {len(resampled_y)})"
        )
        return self

    def predict(self, X:pd.DataFrame | np.ndarray):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X:pd.DataFrame | np.mdarray):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def _apply_resampling(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if self.strategy == "none":
            return X, y

        elif self.strategy == "oversample":
            try:
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(random_state=self.random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                return X_resampled, y_resampled
            except ImportError:
                logger.warning("imblearn not installed, falling back to no resampling")
                return X, y

        elif self.strategy == "undersample":
            try:
                from imblearn.under_sampling import RandomUnderSampler

                rus = RandomUnderSampler(random_state=self.random_state)
                X_resampled, y_resampled = rus.fit_resample(X, y)
                return X_resampled, y_resampled
            except ImportError:
                logger.warning("imblearn not installed, falling back to no resampling")
                return X, y

        elif self.strategy == "class_weight":
            # For class_weight, we don't resample, just pass through
            # The estimator should handle class_weight parameter
            return X, y

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
