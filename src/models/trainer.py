from typing import Dict, List
import mlflow
import mlflow.sklearn
import numpy as np
from joblib import dump
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data.data_loader import load_train_data
from src.data.data_preprocessor import build_preprocessor, save_preprocessor
from src.models.model import get_model
from src.utils.logger import get_logger


log = get_logger("trainer")


def train() -> Dict[str, float]:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X, y = load_train_data()
    preprocessor = build_preprocessor(X)
    model = get_model(cfg.model.type, cfg.model.params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    with mlflow.start_run(run_name=f"{cfg.model.type}"):
        mlflow.log_params({"model_type": cfg.model.type, **cfg.model.params})
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        r2 = float(r2_score(y, preds))

        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        dump(pipeline, "models/model_pipeline.pkl")
        save_preprocessor(preprocessor, "models/preprocessor.pkl")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        log.info("training_completed", rmse=rmse, mae=mae, r2=r2)
        return {"rmse": rmse, "mae": mae, "r2": r2}


def train_all(models: List[str] | None = None) -> Dict[str, Dict[str, float]]:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X, y = load_train_data()
    preprocessor = build_preprocessor(X)
    save_preprocessor(preprocessor, "models/preprocessor.pkl")

    default_params = {
        "linear_regression": {},
        "random_forest": cfg.model.params or {"n_estimators": 200, "random_state": 42},
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
        },
    }

    chosen = models or ["linear_regression", "random_forest", "xgboost"]
    results: Dict[str, Dict[str, float]] = {}

    for m in chosen:
        model = get_model(m, default_params.get(m, {}))
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        with mlflow.start_run(run_name=m):
            mlflow.log_params({"model_type": m, **default_params.get(m, {})})
            pipeline.fit(X, y)

            preds = pipeline.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y, preds)))
            mae = float(mean_absolute_error(y, preds))
            r2 = float(r2_score(y, preds))

            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

            out_path = f"models/model_pipeline_{m}.pkl"
            dump(pipeline, out_path)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            log.info("training_completed", model=m, rmse=rmse, mae=mae, r2=r2)
            results[m] = {"rmse": rmse, "mae": mae, "r2": r2}

    return results