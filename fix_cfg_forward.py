import pathlib
import re

f = pathlib.Path("models/cfg_ddpm.py")
c = f.read_text(encoding="utf-8")

old = """        # Encoder
        h = self.conv_in(x)
        skips = [h]
        enc_idx = 0
        for i, block in enumerate(self.encoder):
            h = block(h, t_emb)
            skips.append(h)
            if enc_idx < len(self.downsample) and (i + 1) % 2 == 0:
                h = self.downsample[enc_idx](h)
                skips.append(h)
                enc_idx += 1
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        # Decoder
        dec_idx = 0
        for i, block in enumerate(self.decoder):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            if dec_idx < len(self.upsample) and (i + 1) % (2 + 1) == 0:
                h = self.upsample[dec_idx](h)
                dec_idx += 1
        return self.conv_out(h)"""

new = """        # Encoder - mirrors __init__ structure exactly
        h = self.conv_in(x)
        skips = []
        enc_idx = 0
        block_idx = 0
        num_levels = len(self.downsample) + 1
        for level in range(num_levels):
            for _ in range(self.num_res_blocks):
                h = self.encoder[block_idx](h, t_emb)
                block_idx += 1
                skips.append(h)
            if enc_idx < len(self.downsample):
                h = self.downsample[enc_idx](h)
                enc_idx += 1
                skips.append(h)
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        # Decoder - mirrors __init__ structure exactly
        dec_idx = 0
        block_idx = 0
        for level in range(num_levels):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder[block_idx](h, t_emb)
                block_idx += 1
            if dec_idx < len(self.upsample):
                h = self.upsample[dec_idx](h)
                dec_idx += 1
        return self.conv_out(h)"""

if old in c:
    c = c.replace(old, new)
    f.write_text(c, encoding="utf-8")
    print("patched")
else:
    print("not found")
