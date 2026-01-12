#!/bin/bash

# Checkpoint Sync Script
# Automatically downloads checkpoints from RunPod every 30 minutes
# Usage: ./sync_checkpoints.sh [interval_minutes]

set -e

# Configuration
REMOTE_HOST="216.81.245.148"
REMOTE_PORT="13030"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_PATH="/workspace/nanochat_experiment/checkpoints/d24/"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/checkpoints"
SYNC_INTERVAL="${1:-30}"  # Default 30 minutes, or pass as first argument

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create local checkpoint directory
mkdir -p "$LOCAL_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Checkpoint Sync Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Remote: ${REMOTE_HOST}:${REMOTE_PORT}"
echo -e "Source: ${REMOTE_PATH}"
echo -e "Local:  ${LOCAL_DIR}"
echo -e "Interval: ${SYNC_INTERVAL} minutes"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to sync checkpoints
sync_checkpoints() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[${timestamp}] Syncing checkpoints...${NC}"
    
    # Check if remote directory exists
    if ! ssh -p "$REMOTE_PORT" -i "$SSH_KEY" -o ConnectTimeout=10 root@"$REMOTE_HOST" "[ -d $REMOTE_PATH ]" 2>/dev/null; then
        echo -e "${RED}✗ Remote directory not found or connection failed${NC}"
        return 1
    fi
    
    # Use rsync to sync only new/changed files
    if rsync -avz --progress \
        -e "ssh -p $REMOTE_PORT -i $SSH_KEY -o ConnectTimeout=10" \
        "root@${REMOTE_HOST}:${REMOTE_PATH}" \
        "$LOCAL_DIR/" 2>&1 | tee /tmp/checkpoint_sync.log; then
        
        # Count checkpoint files
        local checkpoint_count=$(ls -1 "$LOCAL_DIR"/model_*.pt 2>/dev/null | wc -l | tr -d ' ')
        local disk_usage=$(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)
        
        echo -e "${GREEN}✓ Sync completed${NC}"
        echo -e "  Checkpoints: ${checkpoint_count}"
        echo -e "  Disk usage: ${disk_usage}"
        
        # Show latest checkpoint
        if [ "$checkpoint_count" -gt 0 ]; then
            local latest=$(ls -t "$LOCAL_DIR"/model_*.pt 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                local basename=$(basename "$latest")
                local size=$(du -h "$latest" | cut -f1)
                echo -e "  Latest: ${basename} (${size})"
            fi
        fi
    else
        echo -e "${RED}✗ Sync failed - check connection${NC}"
        return 1
    fi
    
    echo ""
}

# Function to handle graceful shutdown
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down checkpoint sync...${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Initial sync
echo -e "${GREEN}Performing initial sync...${NC}"
sync_checkpoints

# Continuous sync loop
echo -e "${BLUE}Starting continuous sync (every ${SYNC_INTERVAL} minutes)${NC}"
echo -e "${BLUE}Press Ctrl+C to stop${NC}"
echo ""

while true; do
    sleep $((SYNC_INTERVAL * 60))
    sync_checkpoints
done
