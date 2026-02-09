import pandas as pd
import pytest

from src.predict.constants import CLIENT_FEATURES_COLUMNS, NEWS_FEATURES_COLUMNS
from src.data import data_loader


class FakeStorage:
    def __init__(self, mapping):
        self.mapping = mapping

    def read_parquet(self, path):
        # match by filename ending
        if path.endswith("X_test.parquet"):
            return self.mapping["X_test"]
        if path.endswith("y_test.parquet"):
            return self.mapping["y_test"]
        if path.endswith("X_train_full.parquet"):
            return self.mapping["X_train_full"]
        raise FileNotFoundError(path)


def test_get_client_features_found_and_missing():
    clients = pd.DataFrame(
        [
            {"userId": "u1", **{c: i for i, c in enumerate(CLIENT_FEATURES_COLUMNS)}},
            {"userId": "u2", **{c: i + 10 for i, c in enumerate(CLIENT_FEATURES_COLUMNS)}},
        ]
    )

    series = data_loader.get_client_features("u1", clients)
    assert series is not None
    assert series["userId"] == "u1"

    none_series = data_loader.get_client_features("unknown", clients)
    assert none_series is None


def test_get_non_viewed_news():
    news = pd.DataFrame({"pageId": ["p1", "p2", "p3"]})
    # Create news features columns with dummy values (one value per row)
    for col in NEWS_FEATURES_COLUMNS:
        news[col] = [0, 1, 2]

    clients = pd.DataFrame({"userId": ["u1", "u1"], "pageId": ["p1", "p2"]})

    result = data_loader.get_non_viewed_news("u1", news, clients)
    assert list(result["pageId"]) == ["p3"]
    assert list(result["userId"]) == ["u1"]


def test_get_predicted_news():
    news = pd.DataFrame({"pageId": ["p1", "p2", "p3"]})
    scores = [10, 50, 40]

    top = data_loader.get_predicted_news(scores, news, n=2, score_threshold=20)
    assert isinstance(top, list)
    assert len(top) == 2
    assert top[0]["pageId"] == "p2"
    assert top[1]["pageId"] == "p3"


def test_get_evaluation_data_and_load_data_for_prediction():
    # Prepare fake X_test and y_test
    X_test = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y_test = pd.Series([0, 1])

    # Prepare full train dataset used for load_data_for_prediction
    full = pd.DataFrame(
        [
            {"pageId": "p1", "userId": "u1", **{c: 0 for c in NEWS_FEATURES_COLUMNS}, **{f: 1 for f in CLIENT_FEATURES_COLUMNS}},
            {"pageId": "p2", "userId": "u2", **{c: 0 for c in NEWS_FEATURES_COLUMNS}, **{f: 1 for f in CLIENT_FEATURES_COLUMNS}},
        ]
    )

    mapping = {"X_test": X_test, "y_test": y_test, "X_train_full": full}
    storage = FakeStorage(mapping)

    eval_df = data_loader.get_evaluation_data(storage=storage)
    assert "TARGET" in eval_df.columns
    assert list(eval_df["TARGET"]) == [0, 1]

    loaded = data_loader.load_data_for_prediction(storage=storage, include_metadata=False)
    assert "news_features" in loaded and "clients_features" in loaded
    assert list(loaded["news_features"]["pageId"]) == ["p1", "p2"]
    assert list(loaded["clients_features"]["userId"]) == ["u1", "u2"]
