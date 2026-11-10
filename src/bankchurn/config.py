"""Configuration management for BankChurn predictor.

All model settings live in ModelConfig.advanced_model, one dictionary per
model name (<name>_params). The factory in models_advanced.py holds the
defaults, and the YAML only overrides what you want to change.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class AdvancedModelConfig(BaseModel):
    """Settings for every model, passed to the factory in models_advanced.py.

    Each field is named <model name>_params and holds a plain dict. Pydantic
    accepts any keys inside the dict, and the factory merges them over its own
    defaults.
    """

    model_config = ConfigDict(extra="forbid")

    logistic_regression_params: dict = Field(default_factory=dict)
    random_forest_params: dict = Field(default_factory=dict)
    gradient_boosting_params: dict = Field(default_factory=dict)
    ensemble_params: dict = Field(default_factory=dict)
    stacking_params: dict = Field(default_factory=dict)
    xgboost_params: dict = Field(default_factory=dict)
    lightgbm_params: dict = Field(default_factory=dict)
    neural_network_params: dict = Field(default_factory=dict)
    mlp_params: dict = Field(default_factory=dict)
    compare_models: List[str] = Field(
        default_factory=list,
        description="Model names to compare. If empty, only 'type' is trained.",
    )


class ModelConfig(BaseModel):
    """Model training configuration."""

    model_config = ConfigDict(extra="forbid")

    type: str = "ensemble"
    test_size: float = Field(0.2, ge=0.0, le=0.4)
    random_state: int = 42
    cv_folds: int = Field(5, ge=2)
    resampling_strategy: str = "none"
    advanced_model: AdvancedModelConfig = AdvancedModelConfig()


class DataConfig(BaseModel):
    """Data preprocessing configuration."""

    model_config = ConfigDict(extra="forbid")

    target_column: str = "Exited"
    categorical_features: List[str] = []
    numerical_features: List[str] = []
    drop_columns: List[str] = []


class MlflowConfig(BaseModel):
    """MLflow tracking configuration."""

    model_config = ConfigDict(extra="forbid")

    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "bankchurn"
    enabled: bool = True


class BankChurnConfig(BaseModel):
    """Complete BankChurn configuration."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    data: DataConfig
    mlflow: MlflowConfig

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> BankChurnConfig:
        """Load and validate the configuration from a YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading config file {config_path}: {e}")
            raise

        if config_dict is None:
            config_dict = {}

        # Fill in missing sections with their defaults
        if "model" not in config_dict:
            config_dict["model"] = ModelConfig().model_dump()
        if "data" not in config_dict:
            config_dict["data"] = DataConfig().model_dump()
        if "mlflow" not in config_dict:
            config_dict["mlflow"] = MlflowConfig().model_dump()

        logger.info(f"Loaded configuration from {config_path}")
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Return the configuration as a plain nested dictionary."""
        return self.model_dump()