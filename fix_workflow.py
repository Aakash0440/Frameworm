import pathlib
f = pathlib.Path('.github/workflows/tests.yml')
c = f.read_text(encoding='utf-8')
c = c.replace(
    'isort --check-only . --skip-glob=".*" --skip ".venv,.git,__pycache__,.egg-info,dist,build"',
    'isort --check-only . --profile black --skip-glob=".*" --skip ".venv,.git,__pycache__,.egg-info,dist,build"'
)
# Also fix duplicate 3.10 in matrix
c = c.replace(
    'python-version: ["3.10", "3.10", "3.11"]',
    'python-version: ["3.10", "3.11", "3.12"]'
)
f.write_text(c, encoding='utf-8')
print("done")
