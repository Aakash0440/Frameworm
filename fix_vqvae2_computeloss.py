import pathlib
f = pathlib.Path('models/vqvae2.py')
c = f.read_text(encoding='utf-8')

insert = '''
    def compute_loss(self, x):
        return self.forward(x)

'''

# Insert before forward method
c = c.replace('    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:', insert + '    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:')
f.write_text(c, encoding='utf-8')
print('done' if 'compute_loss' in c else 'FAILED')
