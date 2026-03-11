import pathlib
f = pathlib.Path('models/vqvae2.py')
c = f.read_text(encoding='utf-8')
c = c.replace('    def compute_loss(self, x):\n        return self.forward(x)\n\n\n', '')
f.write_text(c, encoding='utf-8')
print('removed duplicate, remaining:', c.count('compute_loss'))
