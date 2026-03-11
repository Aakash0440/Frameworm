import pathlib
import re

f = pathlib.Path("search/grid_search.py")
c = f.read_text(encoding="utf-8")

# Show the _evaluate_configuration call to train_fn
idx = c.find("def _evaluate_configuration")
print(c[idx : idx + 800])
