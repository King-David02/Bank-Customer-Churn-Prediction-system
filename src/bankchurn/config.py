from __future__ import annotations
from pydantic import BaseModel, Field
import logging
from pathlib import Path
from typing import Any, List
import yaml

logger = logging.getLogger(__name__)

class LogisticRegressionConfig(BaseModel):
    "Logistic Regression Hyperparameters"
    
    c: float = 0.1
    class_weight: str = "balanced"
    solver: str = "liblinear"
    max_iter: int = 1000

class RandomForestConfig(BaseModel):
    "Random Forest Hyperparameter"
    
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int =5
    max_features: str = "sqrt"

class EnsembleConfig(BaseModel):
    "Ensemble Model congig"
    
    voting:str = Field("soft", pattern="^hard | soft$")
    weight: List[float] = [0.6, 0.4] 

class AdvancedModelConfig(BaseModel):
    "Config for advanced model"
    
    xgboost_param: dict = Field(default_factory=dict)
    lightgbm_params : dict = Field(default_factory=dict)
    neural_networks_params: dict = Field(default_factory=dict)
    mlp_params: dict = Field(default_factory=dict)
    compare_model: List[str] = Field(default_factory=list, description="Models to compare")
    
class ModelConfig(BaseModel):
    "Config for training"
    
    type: str = "logistc_regression"
    test_size: float = Field(0.2, ge=0.0, le=0.4)
    random_state: int = 42
    cv_folds: int = Field(5, ge=2)
    resampling_strategy: str = "none"
    
    #model specific config
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    advanced_model: AdvancedModelConfig = AdvancedModelConfig()

class DataConfig(BaseModel):
    "Data config"
    
    target_column: str = "Exited"
    categorical_features: List[str] = []
    numerical_features: List[str] = []
    drop_columns: List[str] = []

class MlflowConfig(BaseModel):
    "mlflow tracking config"
    
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "bankchurn"
    enabled: bool = True

class BankChurnConfig(BaseModel):
    "complete bank churn config"
    
    model: ModelConfig
    data: DataConfig
    mlflow: MlflowConfig
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> BankChurnConfig:
        config_path=Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"config file not found in {config_path}")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
                
        except yaml.YAMLError as e:
            logger.error(f"failed to parse yaml {config_path}: {e}")
            raise
        
        except Exception as e:
            logger.error(f"error reading {config_path}: {e}")
            raise
        
        if config_dict is None:
            config_dict = {}
            
        if "model" not in config_dict:
            config_dict["model"] = ModelConfig().model_dump()
        if "data" not in config_dict:
            config_dict["data"] = DataConfig().model_dump()
        if "mlflow" not in config_dict:
            config_dict["mlflow"] = MlflowConfig().model_dump()
            
        logger.info(f"Loaded configuration from config_dict")
        return cls(**config_dict)