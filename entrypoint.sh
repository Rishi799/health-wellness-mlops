#!/bin/sh
# entrypoint.sh — prepare the logs directory before starting the API.
#
# When /app/logs is a host bind-mount it may arrive as root-owned and empty.
# Because the container runs as non-root (appuser), we attempt to create a
# test file to verify write access.  If that fails we emit a warning but
# still start the server — PredictionLogger already handles write errors
# gracefully so the API remains functional.
set -e

LOG_DIR="${PREDICTION_LOG_DIR:-/app/logs}"

mkdir -p "$LOG_DIR" 2>/dev/null || true

if touch "$LOG_DIR/.write-test" 2>/dev/null; then
    rm -f "$LOG_DIR/.write-test"
else
    echo "WARNING: $LOG_DIR is not writable by $(whoami). Prediction logs will be skipped." >&2
fi

exec "$@"
