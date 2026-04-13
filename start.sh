#!/bin/bash
set -e

# Generate synthetic data if not present (CSV is gitignored)
if [ ! -f data/historical.csv ]; then
    echo "Generating synthetic customer data..."
    python src/customer_generator.py
fi

# Launch gunicorn with the Flask app
exec gunicorn src.api_server:app \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --threads 4 \
    --timeout 120
