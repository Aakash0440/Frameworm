import pathlib
f = pathlib.Path('models/vqvae2.py')
c = f.read_text(encoding='utf-8')

old = "        top_upsampled = self.top_to_bottom(z_q_top)\n\n        # Concatenate along channel dimension\n        dec_input = torch.cat([z_q_bottom, top_upsampled], dim=1)\n\n        return torch.tanh(self.dec_bottom(dec_input))"
new = "        top_upsampled = self.top_to_bottom(z_q_top)\n        if top_upsampled.shape[2:] != z_q_bottom.shape[2:]:\n            top_upsampled = F.interpolate(top_upsampled, size=z_q_bottom.shape[2:], mode='nearest')\n        dec_input = torch.cat([z_q_bottom, top_upsampled], dim=1)\n        out = torch.tanh(self.dec_bottom(dec_input))\n        if target_size is not None:\n            out = F.interpolate(out, size=tuple(target_size), mode='bilinear', align_corners=False)\n        return out"

if old in c:
    c = c.replace(old, new)
    f.write_text(c, encoding='utf-8')
    print('patched OK')
else:
    print('still not matching')
