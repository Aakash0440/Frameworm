"""
Create animated GIF demos for documentation.

Uses termtosvg to record terminal sessions.
"""

import subprocess
from pathlib import Path


def create_training_demo():
    """Record training session GIF"""
    
    script = """
# Quick training demo
frameworm train --config examples/mnist_vae/config.yaml --epochs 3

# Show progress
echo "✓ Training complete!"
"""
    
    output = Path('docs/assets/training_demo.svg')
    
    # Record with termtosvg
    subprocess.run([
        'termtosvg',
        'record',
        str(output),
        '-c', script,
        '-t', 'window_frame'
    ])
    
    print(f"✓ Created {output}")


def create_cli_demo():
    """Record CLI usage GIF"""
    
    script = """
# Show CLI capabilities
frameworm --help

# List experiments
frameworm experiment list

# Load a plugin
frameworm plugins list
"""
    
    output = Path('docs/assets/cli_demo.svg')
    
    subprocess.run([
        'termtosvg',
        'record',
        str(output),
        '-c', script
    ])
    
    print(f"✓ Created {output}")


if __name__ == '__main__':
    create_training_demo()
    create_cli_demo()