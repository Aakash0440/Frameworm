import pathlib
f = pathlib.Path('tests/unit/test_vqvae2.py')
c = f.read_text(encoding='utf-8')
c = c.replace(
    'from models.vqvae2 import VQVAE2 as _VQVAE2  # noqa: F401 - triggers @register_model',
    'from models.vqvae2 import VQVAE2 as _VQVAE2  # noqa: F401 - triggers @register_model  # isort: skip'
)
f.write_text(c, encoding='utf-8')
print("done")
