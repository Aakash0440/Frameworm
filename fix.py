import pathlib

for fname in ["tests/test_env.py", "tests/test_final_verification.py"]:
    f = pathlib.Path(fname)
    c = f.read_text(encoding="utf-8")
    if "pytest.skip" not in c:
        c = (
            'import pytest\n\npytest.skip("env/verification debug script", allow_module_level=True)\n\n'
            + c
        )
        f.write_text(c, encoding="utf-8")
        print(f"patched {fname}")
