#!/usr/bin/env bash
#
# Script: start_training.sh
# Description: Starts Arabic Piper fine-tuning in the background via nohup,
# so training continues safely when closing an SSH connection.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

LOGS_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGS_DIR"

PID_FILE="$LOGS_DIR/training.pid"

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    EXISTING_PID=$(cat "$PID_FILE")
    if ps -p "$EXISTING_PID" > /dev/null 2>&1; then
        echo "⚠️  Training process is ALREADY RUNNING in background with PID $EXISTING_PID."
        echo "   Use './scripts/status_training.sh' to check progress or './scripts/stop_training.sh' to stop it."
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="$LOGS_DIR/training_${TIMESTAMP}.log"
LATEST_LOG_LINK="$LOGS_DIR/latest.log"

echo "=========================================="
echo "🚀 Starting Arabic Piper Fine-Tuning in Background"
echo "   Log File: $LOG_FILE"
echo "=========================================="

# Activate virtualenv if present in project root or venv folder
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Run run_local_pipeline.py with nohup in background
nohup python3 "$PROJECT_ROOT/scripts/run_local_pipeline.py" "$@" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "$TRAIN_PID" > "$PID_FILE"
ln -sf "$LOG_FILE" "$LATEST_LOG_LINK"

echo "✅ Fine-tuning started in background!"
echo "   PID      : $TRAIN_PID"
echo "   PID File : $PID_FILE"
echo "   Log File : $LOG_FILE"
echo ""
echo "💡 Useful SSH Commands:"
echo "   - View status / GPU:  ./scripts/status_training.sh"
echo "   - Live tail logs   :  tail -f logs/latest.log"
echo "   - Stop gracefully  :  ./scripts/stop_training.sh"
echo ""
echo "🔒 You can now safely close your SSH terminal connection!"
