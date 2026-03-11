import pathlib

f = pathlib.Path("models/gan/dcgan.py")
c = f.read_text(encoding="utf-8")

old = """    def __init__(self, ndf, channels):
        super().__init__()

        self.main = nn.Sequential(
            # Input: channels x 64 x 64
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # Output: 1 x 1 x 1
        )"""

new = """    def __init__(self, ndf, channels, image_size=64):
        super().__init__()
        layers = [nn.Conv2d(channels, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)]
        size = image_size // 2
        ch = ndf
        while size > 4:
            layers += [nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, inplace=True)]
            ch *= 2
            size //= 2
        layers += [nn.Conv2d(ch, 1, 4, 1, 0, bias=False), nn.Sigmoid()]
        self.main = nn.Sequential(*layers)"""

if old in c:
    c = c.replace(old, new)
    # Also update Discriminator instantiation to pass image_size
    c = c.replace(
        "self.discriminator = Discriminator(ndf, channels)",
        'self.discriminator = Discriminator(ndf, channels, image_size=getattr(config.model, "image_size", 64))',
    )
    f.write_text(c, encoding="utf-8")
    print("dcgan patched")
else:
    print("pattern not found")
