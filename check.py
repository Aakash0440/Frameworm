import pathlib

f = pathlib.Path("tests/integration/test_complete_workflow.py")
c = f.read_text(encoding="utf-8")
idx = c.find("deployment.server")
print(repr(c[idx - 10 : idx + 200]))
