#!/bin/bash

# Docker Cleanup Script
# Cleans up Docker system to reduce build times and free up space

set -e

echo "🧹 Starting Docker cleanup..."

# Clean build cache
echo "Cleaning build cache..."
docker builder prune --all --force

# Clean unused images
echo "Cleaning unused images..."
docker image prune --force

# Clean unused volumes
echo "Cleaning unused volumes..."
docker volume prune --force

# Clean unused networks
echo "Cleaning unused networks..."
docker network prune --force

# Show system usage after cleanup
echo "📊 Docker system usage after cleanup:"
docker system df

echo "✅ Docker cleanup completed!"
