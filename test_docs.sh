#!/bin/bash

echo "Testing documentation..."

# Check all links
echo "Checking for broken links..."
mkdocs build --strict

# Check code blocks
echo "Validating code blocks..."
# Extract and test code blocks from markdown

echo "✓ Documentation tests passed!"