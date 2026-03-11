import pathlib
import re

f = pathlib.Path("models/cfg_ddpm.py")
c = f.read_text(encoding="utf-8")

# Add a helper function after imports
helper = '''
def _safe_groups(num_groups: int, channels: int) -> int:
    """Return largest divisor of channels that is <= num_groups."""
    for g in range(min(num_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1

'''

# Insert helper before first class definition
c = re.sub(r"(^class \w)", helper + r"\1", c, count=1, flags=re.MULTILINE)

# Replace all nn.GroupNorm(num_groups, X) with safe version
c = c.replace(
    "self.norm1 = nn.GroupNorm(num_groups, in_channels)",
    "self.norm1 = nn.GroupNorm(_safe_groups(num_groups, in_channels), in_channels)",
)
c = c.replace(
    "self.norm2 = nn.GroupNorm(num_groups, out_channels)",
    "self.norm2 = nn.GroupNorm(_safe_groups(num_groups, out_channels), out_channels)",
)
c = c.replace(
    "self.norm = nn.GroupNorm(num_groups, channels)",
    "self.norm = nn.GroupNorm(_safe_groups(num_groups, channels), channels)",
)
# Fix the hardcoded GroupNorm(8, in_ch) in conv_out
c = c.replace("nn.GroupNorm(8, in_ch)", "nn.GroupNorm(_safe_groups(8, in_ch), in_ch)")

f.write_text(c, encoding="utf-8")
print("cfg_ddpm patched")
