with open('training/trainer.py') as f:
    content = f.read()
content = content.replace("'step': self.global_step,", "'step': self.state.global_step,")
with open('training/trainer.py', 'w') as f:
    f.write(content)
print('Fixed')
