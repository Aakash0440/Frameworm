import pathlib

f = pathlib.Path(".github/workflows/tests.yml")
c = f.read_text(encoding="utf-8")
c = c.replace(
    "pytest tests/unit/ -v --tb=short --cov=frameworm",
    "pytest tests/unit/ -v --tb=short --cov=. --cov-report=xml",
)
f.write_text(c, encoding="utf-8")
print("patched")
print(c[c.find("Run unit") : c.find("Run unit") + 100])
