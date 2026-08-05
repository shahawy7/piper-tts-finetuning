#!/usr/bin/env bash
#
# Script: stop_training.sh
# Description: Gracefully terminates the background training process (SIGINT/SIGTERM)
# to allow PyTorch Lightning to save a final checkpoint cleanly.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PID_FILE="$PROJECT_ROOT/logs/training.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "ℹ️  No active background training PID file found."
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "⏹️  Sending SIGINT (interrupt) signal to PID $PID for graceful checkpoint save..."
    kill -SIGINT "$PID" 2>/dev/null || kill -SIGTERM "$PID" 2>/dev/null
    
    echo "Waiting up to 15 seconds for process to exit gracefully..."
    for i in {1..15}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo "✅ Background training process $PID stopped cleanly."
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
    done

    echo "⚠️ Process did not exit after SIGINT, sending SIGKILL..."
    kill -9 "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "🛑 Process killed."
else
    echo "ℹ️ Process PID $PID is not running."
    rm -f "$PID_FILE"
fi
