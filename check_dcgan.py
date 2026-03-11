import pathlib

f = pathlib.Path("models/gan/dcgan.py")
c = f.read_text(encoding="utf-8")

# The dynamic discriminator still adds stride-2 layers until size<=4,
# but 32 -> 16 -> 8 -> 4, stops. Final conv(ch,1,4,1,0) on 4x4 = 1x1. Fine.
# BUT: 32//2=16 (loop: 16>4, add layer, ch*=2, size=8)
# 8>4: add layer, ch*=4, size=4. Loop exits. Final conv on 4x4. Should work.
# Check the actual loop:
import re

m = re.search(
    r"def __init__\(self, ndf, channels.*?self\.main = nn\.Sequential\(\*layers\)", c, re.DOTALL
)
if m:
    print("Current discriminator __init__:")
    print(m.group())
else:
    print("not found - searching for Discriminator")
    idx = c.find("class Discriminator")
    print(c[idx : idx + 600])
