# Fix load_config missing from core.config
with open("core/config.py") as f:
    content = f.read()
if "load_config" not in content:
    content += """
def load_config(path):
    return Config(path)
"""
    with open("core/config.py", "w") as f:
        f.write(content)
    print("Fixed load_config")
else:
    print("Already exists")
