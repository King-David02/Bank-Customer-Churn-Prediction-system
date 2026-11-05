from __future__ import annotations
import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        create_interactions: bool = True,
        create_ratios: bool = True,
        create_bins: bool = True,
        create_risk_score: bool = True
    ):
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_bins = create_bins
        self.create_risk_score = create_risk_score
        
    def fit(self, X: pd.DataFrame, y=None) -> ChurnFeatureEngineer:
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        
        X_new = X.copy()
        n_original = X.shape[1]
        
        if self.create_interactions:
            X_new = self._add_interactions(X_new)
        if self.create_ratios:
            X_new = self._add_ratios(X_new)
        if self.create_bins:
            X_new = self._add_bins(X_new)
        if self.create_risk_score:
            X_new = self._add_credit(X_new)
        n_new = X_new.shape[1] - n_original
        logger.info(f"Feature engineering: {n_original} → {X.shape[1]} features (+{n_new})")
        return X_new
    
    
    @staticmethod
    def _add_interactions(X: pd.DataFrame) -> pd.DataFrame:
        
        if "Age" in X.columns and "NumOfProducts" in X.columns:
            X["Age_x_NumProducts"] = X["Age"] * X["NumOfProducts"]

        if "Balance" in X.columns and "IsActiveMember" in X.columns:
            X["Balance_x_Active"] = X["Balance"] * X["IsActiveMember"]

        if "CreditScore" in X.columns and "Tenure" in X.columns:
            X["CreditScore_x_Tenure"] = X["CreditScore"] * X["Tenure"]

        if "Age" in X.columns and "IsActiveMember" in X.columns:
            X["Age_x_Active"] = X["Age"] * X["IsActiveMember"]

        return X
    
    @staticmethod
    def _add_ratios(X: pd.DataFrame) -> pd.DataFrame:
        
        if "Balance" in X.columns and "EstimatedSalary" in X.columns:
            X["Balance_Salary_Ratio"] = X["Balance"] / (X["EstimatedSalary"] + 1)

        if "CreditScore" in X.columns and "Age" in X.columns:
            X["CreditScore_Age_Ratio"] = X["CreditScore"] / (X["Age"] + 1)

        if "Balance" in X.columns and "NumOfProducts" in X.columns:
            X["Balance_per_Product"] = X["Balance"] / (X["NumOfProducts"] + 1)

        return X
    
    @staticmethod
    def _add_bins(X: pd.DataFrame) -> pd.DataFrame:
        
        if "Age" in X.columns:
            X["Age_Group"] = pd.cut(
                X["Age"],
                bins=[0, 30, 40, 50, 60, 100],
                labels=["Young", "Early_Mid", "Mid", "Senior", "Elder"],
            ).astype(str)

        if "Balance" in X.columns:
            X["Balance_Band"] = pd.cut(
                X["Balance"],
                bins=[-1, 1, 50000, 100000, 150000, float("inf")],
                labels=["Zero", "Low", "Medium", "High", "VeryHigh"],
            ).astype(str)

        if "Tenure" in X.columns:
            X["Tenure_Band"] = pd.cut(
                X["Tenure"],
                bins=[-1, 2, 5, 8, 11],
                labels=["New", "Growing", "Mature", "Veteran"],
            ).astype(str)

        if "CreditScore" in X.columns:
            X["Credit_Band"] = pd.cut(
                X["CreditScore"],
                bins=[0, 580, 670, 740, 800, 900],
                labels=["Poor", "Fair", "Good", "VeryGood", "Excellent"],
            ).astype(str)

        return X
    
    @staticmethod
    def _add_credit(X: pd.DataFrame) -> pd.DataFrame:
        
        engagement_cols = ["IsActiveMember", "NumOfProducts", "Tenure", "HasCrCard"]
        available = [c for c in engagement_cols if c in X.columns]
        if len(available) >= 2:
            # Normalize each component to 0-1 range before combining
            score = pd.DataFrame(index=X.index)
            if "IsActiveMember" in X.columns:
                score["active"] = X["IsActiveMember"]
            if "NumOfProducts" in X.columns:
                score["products"] = X["NumOfProducts"] / 4.0  # max 4 products
            if "Tenure" in X.columns:
                score["tenure"] = X["Tenure"] / 10.0  # max 10 years
            if "HasCrCard" in X.columns:
                score["card"] = X["HasCrCard"]
            X["Engagement_Score"] = score.mean(axis=1)

        # Financial risk score (higher = more risk of leaving)
        if all(c in X.columns for c in ["Balance", "NumOfProducts"]):
            bal_norm = X["Balance"] / (X["Balance"].max() + 1)
            prod_inv = 1 - (X["NumOfProducts"] / 4.0)
            X["Financial_Risk"] = (bal_norm + prod_inv) / 2

        # Zero-balance flag (strong churn signal)
        if "Balance" in X.columns:
            X["Is_Zero_Balance"] = (X["Balance"] == 0).astype(int)

        # Single-product flag (high churn risk)
        if "NumOfProducts" in X.columns:
            X["Is_Single_Product"] = (X["NumOfProducts"] == 1).astype(int)

        return X