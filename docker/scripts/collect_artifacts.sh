#!/bin/bash
set -e

# Configure MinIO client alias
echo "[$(date)] Configuring MinIO client..."
mc alias set minio "http://${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}"

# Verify bucket exists, create if needed
echo "[$(date)] Ensuring bucket exists: ${MINIO_BUCKET}"
mc mb "minio/${MINIO_BUCKET}" --ignore-existing || true

# Upload all workspace files
echo "[$(date)] Uploading workspace artifacts..."
UPLOAD_COUNT=0

find /workspace -type f | while read file; do
    # Skip the script files themselves
    if [[ "$file" == *.sh ]]; then
        continue
    fi

    filename=$(basename "$file")
    object_path="${MINIO_BUCKET}/${EXECUTION_ID}/${filename}"

    echo "[$(date)] Uploading: ${filename} -> ${object_path}"
    mc cp "$file" "minio/${object_path}"

    UPLOAD_COUNT=$((UPLOAD_COUNT + 1))
done

echo "[$(date)] Artifact collection complete"
