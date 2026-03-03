from core.exceptions import DimensionMismatchError

raise DimensionMismatchError(
    expected_shape=(4, 100, 1, 1),
    received_shape=(4, 100),
    location="models/dcgan.py",
    line=142,
    fix="x.unsqueeze(-1).unsqueeze(-1)"
)