import pandas as pd
from src.data.data_preprocessor import build_preprocessor


def test_build_preprocessor_shapes():
    df = pd.DataFrame({"num1": [1, 2, 3], "cat1": ["a", "b", "a"]})
    pre = build_preprocessor(df)
    X = pre.fit_transform(df)
    assert X.shape[0] == 3
    assert X.shape[1] >= 2