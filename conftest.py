import sys
from unittest.mock import MagicMock

# Block the root __init__.py from loading the training framework during tests.
for mod in [
    "models",
    "training",
    "distributed",
    "search",
    "search.analysis",
    "search.grid_search",
    "search.early_stopping",
    "training.callbacks",
    "distributed.trainer",
    "joblib",
    "matplotlib",
    "matplotlib.pyplot",
]:
    sys.modules.setdefault(mod, MagicMock())

collect_ignore = ["__init__.py"]
