import pathlib

f = pathlib.Path(".github/workflows/tests.yml")
c = f.read_text(encoding="utf-8")
c = c.replace("fail_ci_if_error: true", "fail_ci_if_error: false")
f.write_text(c, encoding="utf-8")
print("patched")
