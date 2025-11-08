from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def get_model(model_type: str = "random_forest", params: dict | None = None):
    params = params or {}
    if model_type == "linear_regression":
        return LinearRegression(**params)
    if model_type == "random_forest":
        return RandomForestRegressor(**params)
    if model_type == "xgboost":
        default = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
        }
        # merge defaults with params, params override defaults
        cfg = {**default, **params}
        return XGBRegressor(**cfg)
    raise ValueError(f"Unsupported model type: {model_type}")