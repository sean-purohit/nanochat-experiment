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
BASE_CHECKPOINT_DIR="$(cd "$(dirname "$0")" && pwd)/checkpoints"
SYNC_INTERVAL="${1:-30}"  # Default 30 minutes, or pass as first argument

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create base checkpoint directory
mkdir -p "$BASE_CHECKPOINT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Checkpoint Sync Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Remote: ${REMOTE_HOST}:${REMOTE_PORT}"
echo -e "Source: ${REMOTE_PATH}"
echo -e "Local:  ${BASE_CHECKPOINT_DIR}/<timestamp>/"
echo -e "Interval: ${SYNC_INTERVAL} minutes"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to sync checkpoints
sync_checkpoints() {
    local display_timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local folder_timestamp=$(date '+%Y-%m-%d_%H-%M')
    echo -e "${YELLOW}[${display_timestamp}] Syncing checkpoints...${NC}"
    
    # Check if remote directory exists
    if ! ssh -p "$REMOTE_PORT" -i "$SSH_KEY" -o ConnectTimeout=10 root@"$REMOTE_HOST" "[ -d $REMOTE_PATH ]" 2>/dev/null; then
        echo -e "${RED}✗ Remote directory not found or connection failed${NC}"
        return 1
    fi
    
    # Get list of remote checkpoint files
    local remote_files=$(ssh -p "$REMOTE_PORT" -i "$SSH_KEY" -o ConnectTimeout=10 root@"$REMOTE_HOST" "ls ${REMOTE_PATH}" 2>/dev/null)
    
    if [ -z "$remote_files" ]; then
        echo -e "${YELLOW}⚠ No checkpoint files found on remote${NC}"
        return 0
    fi
    
    # Create timestamped directory for this sync
    local sync_dir="$BASE_CHECKPOINT_DIR/$folder_timestamp"
    mkdir -p "$sync_dir"
    echo -e "  Saving to: ${BLUE}${folder_timestamp}/${NC}"
    echo ""
    
    local files_downloaded=0
    local files_skipped=0
    local total_files=0
    
    # Download each file
    while IFS= read -r file; do
        total_files=$((total_files + 1))
        local remote_file="${REMOTE_PATH}${file}"
        local local_file="${sync_dir}/${file}"
        
        if [ -f "$local_file" ]; then
            echo -e "  ${BLUE}Skip${NC} ${file} (already exists)"
            files_skipped=$((files_skipped + 1))
        else
            echo -e "  ${GREEN}Download${NC} ${file}..."
            if scp -P "$REMOTE_PORT" -i "$SSH_KEY" -o ConnectTimeout=10 \
                "root@${REMOTE_HOST}:${remote_file}" "$local_file" 2>&1 | grep -v "Bytes"; then
                files_downloaded=$((files_downloaded + 1))
                echo -e "    ${GREEN}✓${NC} Downloaded"
            else
                echo -e "    ${RED}✗${NC} Failed"
            fi
        fi
    done <<< "$remote_files"
    
    # Count checkpoint files in current sync
    local checkpoint_count=$(ls -1 "${sync_dir}"/model_*.pt 2>/dev/null | wc -l | tr -d ' ')
    local sync_disk_usage=$(du -sh "$sync_dir" 2>/dev/null | cut -f1)
    local total_disk_usage=$(du -sh "$BASE_CHECKPOINT_DIR" 2>/dev/null | cut -f1)
    
    echo ""
    echo -e "${GREEN}✓ Sync completed${NC}"
    echo -e "  Location: ${folder_timestamp}/"
    echo -e "  Downloaded: ${files_downloaded} files"
    echo -e "  Skipped: ${files_skipped} files"
    echo -e "  Total files: ${total_files}"
    echo -e "  Checkpoints: ${checkpoint_count}"
    echo -e "  This sync: ${sync_disk_usage}"
    echo -e "  Total usage: ${total_disk_usage}"
    
    # Show checkpoint step numbers in this sync
    if [ "$checkpoint_count" -gt 0 ]; then
        echo -e "  Steps: $(ls "${sync_dir}"/model_*.pt 2>/dev/null | sed 's/.*model_0*//' | sed 's/.pt//' | tr '\n' ', ' | sed 's/,$//')"
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
