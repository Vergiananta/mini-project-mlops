from typing import Optional
import yaml
from pydantic import BaseModel


class MLflowConfig(BaseModel):
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "house-price"


class DataConfig(BaseModel):
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    target_column: str = "SalePrice"


class ModelConfig(BaseModel):
    type: str = "random_forest"  # options: linear_regression, random_forest
    params: dict = {"n_estimators": 200, "random_state": 42}


class AppConfig(BaseModel):
    log_level: str = "INFO"
    mlflow: MLflowConfig = MLflowConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()


def load_config(path: str = "config/config.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}
    return AppConfig(
        log_level=cfg_dict.get("log_level", "INFO"),
        mlflow=MLflowConfig(**cfg_dict.get("mlflow", {})),
        data=DataConfig(**cfg_dict.get("data", {})),
        model=ModelConfig(**cfg_dict.get("model", {})),
    )