import pathlib
import re

f = pathlib.Path("models/cfg_ddpm.py")
c = f.read_text(encoding="utf-8")

# Fix 1: store num_res_blocks
c = c.replace(
    "self.num_classes = num_classes",
    "self.num_classes = num_classes\n        self.num_res_blocks = num_res_blocks",
)

# Fix 2: replace forward with correct skip logic
# Find the encoder/decoder section by line numbers and replace
lines = c.splitlines()
start = next(i for i, l in enumerate(lines) if "# Encoder" in l and "conv_in" in lines[i + 1])
end = next(i for i, l in enumerate(lines) if "return self.conv_out(h)" in l and i > start) + 1

new_forward = """        # Encoder - mirrors __init__ structure exactly
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

lines[start:end] = new_forward.splitlines()
f.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("cfg_ddpm forward patched, start line:", start)
