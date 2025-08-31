#!/bin/bash
set -e

echo "📥 Downloading weights..."
mkdir -p /app/models
gdown --id "$GDRIVE_ID" -O /app/models/weights.zip

echo "📂 Extracting weights..."
unzip -o /app/models/weights.zip -d /app/models

rm /app/models/weights.zip
echo "✅ Weights ready in /app/models"

exec "$@"
