#!/bin/bash

echo "Testing FRAMEWORM CLI"
echo "===================="

# Test help
echo "Testing help..."
python -m frameworm --help
python -m frameworm train --help
python -m frameworm search --help

# Test init
echo "Testing init..."
python -m frameworm init test-project --template vae --path /tmp
ls /tmp/test-project

# Test config
echo "Testing config..."
python -m frameworm config validate /tmp/test-project/configs/config.yaml

echo "===================="
echo "✓ CLI tests complete"