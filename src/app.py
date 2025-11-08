from typing import Any, Dict, List
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from joblib import load

from prometheus_fastapi_instrumentator import Instrumentator
from src.utils.logger import configure_logging, get_logger
from src.config import load_config
import io
import datetime
import mlflow


configure_logging()
log = get_logger("api")
cfg = load_config()

app = FastAPI(title="House Price Prediction API", version="0.1.0")
Instrumentator().instrument(app).expose(app)

# Inisialisasi MLflow
try:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
except Exception as e:
    log.warning("mlflow_init_failed", error=str(e))


class PredictionItem(BaseModel):
    row: int
    prediction: float


class PredictFileResponse(BaseModel):
    model: str
    rows: int
    predictions: List[PredictionItem]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def log_api_error_mlflow(request: Request, status_code: int, detail: Any) -> None:
    try:
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        with mlflow.start_run(run_name=f"api_error_{status_code}"):
            mlflow.set_tag("source", "fastapi")
            mlflow.set_tag("endpoint", request.url.path)
            mlflow.log_params(
                {
                    "status_code": status_code,
                    "method": request.method,
                    "path": request.url.path,
                    "client": (request.client.host if request.client else "unknown"),
                    "model": str(request.query_params.get("model", "")),
                }
            )
            mlflow.log_dict(
                {
                    "timestamp": ts,
                    "detail": detail if isinstance(detail, str) else str(detail),
                    "query_params": dict(request.query_params),
                },
                "error.json",
            )
    except Exception as e:
        log.error("mlflow_log_error_failed", error=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if 400 <= exc.status_code < 600:
        log_api_error_mlflow(request, exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_api_error_mlflow(request, 500, str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.post("/predict", response_model=PredictFileResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Query("random_forest", description="Pilih model: random_forest, linear_regression, xgboost"),
) -> PredictFileResponse:
    fname = file.filename.lower()
    try:
        contents = await file.read()
        if fname.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        elif fname.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="File harus berformat .csv atau .xlsx")
        log.info("predict_file_received", filename=file.filename, rows=len(df), cols=len(df.columns))
    except HTTPException:
        raise
    except Exception as e:
        log.error("file_read_failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Gagal membaca file: {e}")
    log.info("predict_model_selected", model=model)
    allowed = {"random_forest", "linear_regression", "xgboost"}
    if model not in allowed:
        raise HTTPException(status_code=400, detail=f"Model tidak dikenal: {model}. Pilih salah satu dari {sorted(allowed)}")

    pipeline_path = f"models/model_pipeline_{model}.pkl"
    try:
        pipeline = load(pipeline_path)
    except Exception as e:
        try:
            pipeline = load("models/model_pipeline.pkl")
            log.warning("model_load_fallback_single", requested=model)
        except Exception:
            log.error("model_load_failed", error=str(e), path=pipeline_path)
            raise HTTPException(status_code=500, detail="Model tidak tersedia. Jalankan training terlebih dahulu.")

    try:
        expected_cols = list(pipeline.named_steps["preprocessor"].feature_names_in_)
    except Exception:
        expected_cols = None

    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in df.columns]
        extras = [c for c in df.columns if c not in expected_cols]
        if missing:
            log.error("predict_missing_columns", missing=missing)
            raise HTTPException(status_code=400, detail=f"Kolom hilang pada file: {missing}")

    try:
        preds = pipeline.predict(df)
        predictions = [PredictionItem(row=i, prediction=float(v)) for i, v in enumerate(preds)]
        log.info("predict_success_batch", count=len(predictions))
        return PredictFileResponse(model=model, rows=len(df), predictions=predictions)
    except Exception as e:
        log.error("predict_failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Prediksi gagal: {e}")