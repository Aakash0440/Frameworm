"""Tests for Bayesian search"""

import pytest

try:
    from search import BayesianSearch
    from search.space import Real, Integer
    from core import Config

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


@pytest.mark.skipif(not BAYESIAN_AVAILABLE, reason="scikit-optimize not installed")
class TestBayesianSearch:
    def test_bayesian_search(self):
        def train_fn(config):
            lr = config.training.lr
            return {"val_loss": abs(lr - 0.0005) * 1000}

        base_config = Config.from_dict({"training": {"lr": 0.001}})
        search_space = {"training.lr": Real(0.0001, 0.001, log=False)}

        search = BayesianSearch(
            base_config,
            search_space,
            metric="val_loss",
            mode="min",
            n_trials=10,
            n_initial_points=3,
            random_state=42,
        )

        best_config, best_score = search.run(train_fn, verbose=False)

        assert len(search.results) == 10
        assert 0.0001 <= best_config["training.lr"] <= 0.001

        # Should find near-optimal (0.0005)
        assert abs(best_config["training.lr"] - 0.0005) < 0.0002


# Run tests
if BAYESIAN_AVAILABLE:
    pytest.main([__file__, "-v"])
else:
    print("Bayesian search tests skipped (scikit-optimize not installed)")
