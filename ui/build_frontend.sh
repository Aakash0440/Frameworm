#!/bin/bash

echo "Building FRAMEWORM Dashboard..."

cd frameworm/ui/frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build production bundle
echo "Building production bundle..."
npm run build

# Copy build to static directory
echo "Copying build files..."
mkdir -p ../static
cp -r build/* ../static/

echo "✓ Dashboard built successfully!"