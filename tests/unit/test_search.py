"""Tests for hyperparameter search"""

import pytest
import numpy as np
from search.space import Categorical, Integer, Real, expand_search_space
from search import GridSearch, RandomSearch, SearchAnalyzer
from core import Config


class TestSearchSpace:
    def test_categorical(self):
        cat = Categorical(["a", "b", "c"])
        value = cat.sample(random_state=42)
        assert value in ["a", "b", "c"]

    def test_integer(self):
        integer = Integer(1, 10, log=False)
        value = integer.sample(random_state=42)
        assert 1 <= value <= 10

    def test_real(self):
        real = Real(0.0, 1.0, log=False)
        value = real.sample(random_state=42)
        assert 0.0 <= value <= 1.0

    def test_expand_search_space(self):
        space = expand_search_space({"param1": [1, 2, 3], "param2": Integer(1, 10)})

        assert isinstance(space["param1"], Categorical)
        assert isinstance(space["param2"], Integer)


class TestGridSearch:
    def test_grid_search(self):
        # Dummy train function
        def train_fn(config):
            lr = config.training.lr
            return {"val_loss": lr * 1000}

        base_config = Config.from_dict({"training": {"lr": 0.001}})
        search_space = {"training.lr": [0.001, 0.0001]}

        search = GridSearch(base_config, search_space, metric="val_loss", mode="min")
        best_config, best_score = search.run(train_fn, verbose=False)

        assert best_config["training.lr"] == 0.0001  # Lowest lr gives lowest loss
        assert len(search.results) == 2


class TestRandomSearch:
    def test_random_search(self):
        def train_fn(config):
            lr = config.training.lr
            return {"val_loss": abs(lr - 0.0005) * 1000}

        base_config = Config.from_dict({"training": {"lr": 0.001}})
        search_space = {"training.lr": Real(0.0001, 0.001, log=False)}

        search = RandomSearch(
            base_config, search_space, metric="val_loss", mode="min", n_trials=10, random_state=42
        )

        best_config, best_score = search.run(train_fn, verbose=False)

        assert len(search.results) == 10
        assert 0.0001 <= best_config["training.lr"] <= 0.001


class TestSearchAnalyzer:
    def test_analyzer(self):
        results = [
            {"trial_idx": i, "config": {"param": i}, "metrics": {}, "score": float(i)}
            for i in range(10)
        ]

        analyzer = SearchAnalyzer(results)

        top3 = analyzer.get_best_n(3)
        assert len(top3) == 3
        assert top3.iloc[0]["score"] == 0.0  # Best score
