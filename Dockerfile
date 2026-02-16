# Multi-stage build for smaller image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set up working directory
WORKDIR /app

# Copy application code
COPY frameworm/ ./frameworm/
COPY models/ ./models/

# Add .local/bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "-m", "frameworm.deployment.server", "--model", "models/model.pt", "--port", "8000"]
EOF

# Create .dockerignore
cat > .dockerignore << 'EOF'
__pycache__
*.pyc
*.pyo
*.pyd
.git
.gitignore
.venv
venv/
*.egg-info
.pytest_cache
.coverage
htmlcov/
dist/
build/
*.log
.DS_Store