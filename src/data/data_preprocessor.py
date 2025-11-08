from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump, load
from src.utils.logger import get_logger


log = get_logger("preprocessor")


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features: List[str] = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features: List[str] = df.select_dtypes(exclude=["number"]).columns.tolist()
    log.info("preprocessor_features_split", numeric=len(numeric_features), categorical=len(categorical_features))

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, path: str = "models/preprocessor.pkl") -> None:
    dump(preprocessor, path)
    log.info("preprocessor_saved", path=path)


def load_preprocessor(path: str = "models/preprocessor.pkl") -> ColumnTransformer:
    pre = load(path)
    log.info("preprocessor_loaded", path=path)
    return pre