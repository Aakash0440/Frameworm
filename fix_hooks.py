with open("plugins/hooks.py") as f:
    content = f.read()

hooks_to_add = [
    "on_backward_begin",
    "on_backward_end", 
    "on_optimizer_step"
]

for hook in hooks_to_add:
    if hook not in content:
        content = content.replace(
            "SUPPORTED_HOOKS = [",
            f'SUPPORTED_HOOKS = [\n    "{hook}",'
        )

with open("plugins/hooks.py", "w") as f:
    f.write(content)
print("Hooks fixed")
