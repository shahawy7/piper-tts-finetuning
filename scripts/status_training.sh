#!/usr/bin/env bash
#
# Script: status_training.sh
# Description: Checks the status of background Arabic Piper fine-tuning,
# displays GPU stats, and prints recent log output.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PID_FILE="$PROJECT_ROOT/logs/training.pid"
LATEST_LOG="$PROJECT_ROOT/logs/latest.log"

echo "=========================================="
echo "📊 Background Training Status"
echo "=========================================="

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "🟢 Status : RUNNING (PID $PID)"
    else
        echo "🔴 Status : STOPPED / COMPLETED (Stale PID $PID)"
    fi
else
    echo "⚪ Status : NO ACTIVE BACKGROUND PROCESS"
fi

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️  GPU Utilization:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
fi

if [ -f "$LATEST_LOG" ]; then
    echo ""
    echo "📋 Recent Log Output (tail -n 20 $LATEST_LOG):"
    echo "--------------------------------------------------"
    tail -n 20 "$LATEST_LOG"
    echo "--------------------------------------------------"
else
    echo "ℹ️  No log file found in logs/"
fi
