import pathlib

f = pathlib.Path("models/cfg_ddpm.py")
c = f.read_text(encoding="utf-8")

old = """            for j in range(num_res_blocks + 1):
                if j < num_res_blocks:
    skip_ch = channels[-(i + 1)]
    elif i < len(channels) - 1:
    skip_ch = channels[-(i + 2)]
    else:
    skip_ch = channels[0]  # conv_in output"""

# Just replace the whole decoder loop cleanly
old2 = """        for i, out_ch in enumerate(reversed(channels)):
            for j in range(num_res_blocks + 1):
                if j < num_res_blocks:
    skip_ch = channels[-(i + 1)]
    elif i < len(channels) - 1:
    skip_ch = channels[-(i + 2)]
    else:
    skip_ch = channels[0]  # conv_in output
                self.decoder.append(
                    ResBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout=dropout)
                )
                in_ch = out_ch"""

new2 = """        for i, out_ch in enumerate(reversed(channels)):
            for j in range(num_res_blocks + 1):
                if j < num_res_blocks:
                    skip_ch = channels[-(i + 1)]
                elif i < len(channels) - 1:
                    skip_ch = channels[-(i + 2)]
                else:
                    skip_ch = channels[0]
                self.decoder.append(
                    ResBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout=dropout)
                )
                in_ch = out_ch"""

if old2 in c:
    c = c.replace(old2, new2)
    f.write_text(c, encoding="utf-8")
    print("patched")
else:
    print("not found, showing current decoder loop:")
    idx = c.find("for i, out_ch in enumerate(reversed(channels)):")
    print(repr(c[idx : idx + 500]))
