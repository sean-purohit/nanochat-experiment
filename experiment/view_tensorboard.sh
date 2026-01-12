#!/bin/bash

# TensorBoard Viewer Script
# Creates SSH tunnel to access TensorBoard running on RunPod
# Usage: ./view_tensorboard.sh

set -e

# Configuration
REMOTE_HOST="216.81.245.148"
REMOTE_PORT="13030"
SSH_KEY="$HOME/.ssh/id_ed25519"
LOCAL_PORT="6006"
REMOTE_TB_PORT="6006"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TensorBoard Viewer${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if port is already in use
if lsof -Pi :$LOCAL_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠ Port $LOCAL_PORT is already in use${NC}"
    echo -e "Checking if it's our tunnel..."
    existing_pid=$(lsof -Pi :$LOCAL_PORT -sTCP:LISTEN -t 2>/dev/null | head -1)
    if ps -p $existing_pid -o command= | grep -q "$REMOTE_HOST"; then
        echo -e "${GREEN}✓ TensorBoard tunnel already running!${NC}"
        echo ""
        echo -e "${GREEN}Open in your browser:${NC}"
        echo -e "  ${BLUE}http://localhost:$LOCAL_PORT${NC}"
        echo ""
        echo -e "Press Ctrl+C to close the existing tunnel and restart"
        sleep 2
        kill $existing_pid 2>/dev/null || true
        sleep 2
    else
        echo -e "${YELLOW}Port is used by another process. Kill it? (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            kill $existing_pid 2>/dev/null || true
            sleep 2
        else
            echo "Exiting..."
            exit 1
        fi
    fi
fi

echo -e "${YELLOW}Starting SSH tunnel...${NC}"
echo -e "Local:  http://localhost:$LOCAL_PORT"
echo -e "Remote: ${REMOTE_HOST}:${REMOTE_PORT}"
echo ""

# Start SSH tunnel
echo -e "${GREEN}✓ Tunnel established!${NC}"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Open TensorBoard in your browser:${NC}"
echo -e ""
echo -e "  ${BLUE}http://localhost:$LOCAL_PORT${NC}"
echo -e ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the tunnel${NC}"
echo ""

# Keep tunnel open
ssh -N -L ${LOCAL_PORT}:localhost:${REMOTE_TB_PORT} \
    -p ${REMOTE_PORT} \
    -i ${SSH_KEY} \
    root@${REMOTE_HOST}
