#!/bin/bash

# Quick start script for the Pacman vs Ghost Arena
# Usage: ./run_game.sh <seeker_id> <hider_id> [arena options]

set -e

if [ "$#" -lt 2 ]; then
	echo "Usage: $0 <seeker_id> <hider_id> [arena options]"
	exit 1
fi

SEEKER="$1"
HIDER="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"

if command -v conda >/dev/null 2>&1; then
	PYTHON_CMD=(conda run -n ml python)
else
	PYTHON_CMD=(python)
fi

cd "$SRC_DIR"

"${PYTHON_CMD[@]}" arena.py --seek "$SEEKER" --hide "$HIDER" "$@"
