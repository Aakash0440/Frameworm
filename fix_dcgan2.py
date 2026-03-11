import pathlib

f = pathlib.Path("models/gan/dcgan.py")
c = f.read_text(encoding="utf-8")
c = c.replace(
    "        real_preds = self.discriminator(real_images)\n        d_loss_real = F.binary_cross_entropy(real_preds, real_labels)",
    "        real_preds = self.discriminator(real_images).view(-1)\n        d_loss_real = F.binary_cross_entropy(real_preds, real_labels)",
)
c = c.replace(
    "        fake_preds = self.discriminator(fake_images.detach())\n        d_loss_fake = F.binary_cross_entropy(fake_preds, fake_labels)",
    "        fake_preds = self.discriminator(fake_images.detach()).view(-1)\n        d_loss_fake = F.binary_cross_entropy(fake_preds, fake_labels)",
)
c = c.replace(
    "        gen_preds = self.discriminator(fake_images)\n        g_loss = F.binary_cross_entropy(gen_preds, real_labels)",
    "        gen_preds = self.discriminator(fake_images).view(-1)\n        g_loss = F.binary_cross_entropy(gen_preds, real_labels)",
)
f.write_text(c, encoding="utf-8")
print("done")
