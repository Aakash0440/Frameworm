import re

with open("models/vqvae2.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: decode method signature
content = re.sub(
    r"def decode\(self, z_q_top: torch\.Tensor, z_q_bottom: torch\.Tensor\) -> torch\.Tensor:",
    "def decode(self, z_q_top: torch.Tensor, z_q_bottom: torch.Tensor, target_size=None) -> torch.Tensor:",
    content,
)

# Fix 2: decode body - find the top_to_bottom block and replace it
old = (
    "top_upsampled = self.top_to_bottom(z_q_top)\n"
    "        # Concatenate along channel dimension\n"
    "        dec_input = torch.cat([z_q_bottom, top_upsampled], dim=1)\n"
    "        return torch.tanh(self.dec_bottom(dec_input))"
)
new = (
    "top_upsampled = self.top_to_bottom(z_q_top)\n"
    "        if top_upsampled.shape[2:] != z_q_bottom.shape[2:]:\n"
    "            top_upsampled = F.interpolate(top_upsampled, size=z_q_bottom.shape[2:], mode='nearest')\n"
    "        dec_input = torch.cat([z_q_bottom, top_upsampled], dim=1)\n"
    "        out = torch.tanh(self.dec_bottom(dec_input))\n"
    "        if target_size is not None:\n"
    "            out = F.interpolate(out, size=tuple(target_size), mode='bilinear', align_corners=False)\n"
    "        return out"
)

if old in content:
    content = content.replace(old, new)
    print("decode body patched")
else:
    # Try without the comment line (in case it was already removed)
    old2 = (
        "top_upsampled = self.top_to_bottom(z_q_top)\n"
        "        dec_input = torch.cat([z_q_bottom, top_upsampled], dim=1)\n"
        "        return torch.tanh(self.dec_bottom(dec_input))"
    )
    if old2 in content:
        content = content.replace(old2, new)
        print("decode body patched (no comment variant)")
    else:
        print("WARNING: decode body pattern not found - check manually")

# Fix 3: forward - pass target_size
old_fwd = "recon = self.decode(z_q_top, z_q_bottom)"
new_fwd = "recon = self.decode(z_q_top, z_q_bottom, target_size=x.shape[2:])"
if old_fwd in content:
    content = content.replace(old_fwd, new_fwd)
    print("forward patched")
else:
    print("WARNING: forward pattern not found")

with open("models/vqvae2.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done - verify with: Select-String -Path models\\vqvae2.py -Pattern 'target_size|interpolate'")