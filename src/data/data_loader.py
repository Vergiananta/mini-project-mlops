from typing import Tuple
import pandas as pd
from src.config import load_config
from src.utils.logger import get_logger


log = get_logger("data_loader")


def load_train_data() -> Tuple[pd.DataFrame, pd.Series]:
    cfg = load_config()
    df = pd.read_csv(cfg.data.train_path)
    log.info("train_data_loaded", rows=len(df), path=cfg.data.train_path)
    if cfg.data.target_column not in df.columns:
        raise ValueError(f"Target column '{cfg.data.target_column}' not found in training data")
    y = df[cfg.data.target_column]
    X = df.drop(columns=[cfg.data.target_column])
    return X, y


def load_test_data() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_csv(cfg.data.test_path)
    log.info("test_data_loaded", rows=len(df), path=cfg.data.test_path)
    return df