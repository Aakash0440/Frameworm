"""
Generate complete API reference from docstrings.

Uses sphinx-autodoc to extract all docstrings and create
comprehensive API documentation.
"""

import os
import subprocess
from pathlib import Path


def generate_api_reference():
    """Generate API reference documentation"""
    
    docs_api_dir = Path('docs/api-reference')
    docs_api_dir.mkdir(exist_ok=True, parents=True)
    
    # Module structure
    modules = {
        'core': ['config', 'registry', 'types'],
        'training': ['trainer', 'callbacks', 'schedulers'],
        'models': ['vae', 'dcgan', 'ddpm', 'vqvae2', 'vitgan', 'cfg_ddpm'],
        'experiment': ['experiment', 'manager', 'visualization'],
        'search': ['grid', 'random', 'bayesian'],
        'metrics': ['fid', 'inception_score', 'lpips'],
        'deployment': ['exporter', 'server', 'quantization'],
        'integrations': ['wandb', 'mlflow', 'storage', 'notifications'],
        'monitoring': ['metrics', 'model_registry', 'drift', 'ab_testing'],
        'production': ['health', 'shutdown', 'rate_limit', 'validation', 'security'],
        'plugins': ['hooks', 'loader'],
        'cli': ['main']
    }
    
    # Generate index
    index_content = """# API Reference

Complete API documentation for all FRAMEWORM modules.

## Core Modules
````{toctree}
:maxdepth: 2

core/index
training/index
models/index
experiment/index
````

## Advanced Modules
````{toctree}
:maxdepth: 2

search/index
metrics/index
deployment/index
integrations/index
monitoring/index
production/index
plugins/index
cli/index
````

## Quick Links

- **Getting Started:** {doc}`../user-guide/quickstart`
- **Tutorials:** {doc}`../tutorials/index`
- **Examples:** [GitHub Examples](https://github.com/yourusername/frameworm/tree/main/examples)
"""
    
    (docs_api_dir / 'index.md').write_text(index_content)
    
    # Generate module pages
    for module_name, submodules in modules.items():
        module_dir = docs_api_dir / module_name
        module_dir.mkdir(exist_ok=True)
        
        # Module index
        module_index = f"""# {module_name.title()} API
````{{toctree}}
:maxdepth: 1

"""
        for submodule in submodules:
            module_index += f"{submodule}\n"
        
        module_index += "```\n"
        (module_dir / 'index.md').write_text(module_index)
        
        # Generate each submodule page
        for submodule in submodules:
            submodule_path = f'frameworm.{module_name}.{submodule}'
            
            content = f"""# {submodule.replace('_', ' ').title()}
```{{eval-rst}}
.. automodule:: {submodule_path}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__
```

## Examples
```python
from {submodule_path} import *

# See user guide for usage examples
```

## See Also

- {{doc}}`../user-guide/index`
- {{doc}}`../tutorials/index`
"""
            
            (module_dir / f'{submodule}.md').write_text(content)
    
    print("✓ API reference generated")


def setup_sphinx():
    """Setup Sphinx configuration for API docs"""
    
    conf_additions = """
# Sphinx autodoc
extensions.extend([
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
])

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autosummary_generate = True
"""
    
    conf_path = Path('docs/conf.py')
    if conf_path.exists():
        content = conf_path.read_text()
        if 'sphinx.ext.autodoc' not in content:
            conf_path.write_text(content + '\n' + conf_additions)
            print("✓ Sphinx configuration updated")


if __name__ == '__main__':
    generate_api_reference()
    setup_sphinx()
    print("\n✓ Run: mkdocs build")
