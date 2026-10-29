#!/bin/sh
# run_chargpt.sh - start a Mojo script in the background with a timestamped log.
#
# Usage:
#   ./run_chargpt.sh                          # runs chargpt.mojo
#   ./run_chargpt.sh max_iters=5000           # extra args go to the program
#   ./run_chargpt.sh other.mojo lr=0.001      # first arg ending in .mojo = target
#
# Each run creates, inside ./run_logs/ (next to this script):
#   <name>_<YYYY-MM-DD_HH-MM-SS>.log   output (stdout + stderr)
#   <name>_<YYYY-MM-DD_HH-MM-SS>.pid   process id
#   <name>_latest.log                  symlink to the newest log (best effort)
#
# Works with any POSIX sh on Linux and macOS. On Windows use WSL or Git Bash.

set -eu

# Always work relative to the script's own folder, wherever it is called from.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# `quick` smoke-test mode: run chargpt.mojo with a small config. Detected
# before -h handling so `./run_chargpt.sh quick ...` works.
QUICK_MODE=0
case "${1:-}" in
  quick)
    QUICK_MODE=1
    shift
    ;;
esac

# -h / --help: print the defaults chargpt.mojo would use, parsed from
# shared.mojo, without compiling anything.
case "${1:-}" in
  -h|--help)
    if ! command -v awk >/dev/null 2>&1; then
      echo "run_chargpt.sh: 'awk' is required to print defaults but was not found." >&2
      exit 1
    fi
    DEFAULTS=$(awk '
      /^[[:space:]]*(batch_size|seq_length|eval_iters|eval_interval|max_iters|max_new_tokens|n_embd|n_head|n_layer|learning_rate|drop_ratio|kv_cache):/ {
        line = $0
        sub(/^[[:space:]]+/, "", line)
        field = line
        sub(/:.*/, "", field)
        val = line
        sub(/.*=[[:space:]]*/, "", val)
        sub(/,$/, "", val)
        printf "%s=%s ", field, val
      }
    ' "$SCRIPT_DIR/shared.mojo")
    if [ -z "$DEFAULTS" ]; then
      echo "run_chargpt.sh: could not parse defaults from shared.mojo" >&2
      exit 1
    fi
    printf '%s\n' "${DEFAULTS% }"
    exit 0
    ;;
esac

# Optional first argument: the .mojo file to run (default chargpt.mojo).
TARGET="chargpt.mojo"
if [ "$QUICK_MODE" = "0" ]; then
    case "${1:-}" in
        *.mojo) TARGET="$1"; shift ;;
    esac
fi
if [ "$QUICK_MODE" = "1" ]; then
    TARGET="chargpt.mojo"
fi

if [ ! -f "$TARGET" ]; then
    echo "error: $TARGET not found in $SCRIPT_DIR" >&2
    exit 1
fi
if ! command -v mojo >/dev/null 2>&1; then
    echo "error: 'mojo' not found on PATH (activate your environment first)" >&2
    exit 1
fi

if [ "$QUICK_MODE" = "1" ]; then
    QUICK_DEFAULTS="batch_size=8 seq_length=24 eval_iters=50 eval_interval=300 max_iters=1000 max_new_tokens=500 n_embd=32 n_head=1 n_layer=1 learning_rate=1e-3 drop_ratio=0.0 kv_cache=false"
    # User args come first so overrides win (CmdLineArgs keeps first occurrence).
    set -- "$@" $QUICK_DEFAULTS
fi

NAME=$(basename "$TARGET" .mojo)
LOG_DIR="run_logs"
mkdir -p "$LOG_DIR"

# Filename-safe human readable timestamp, e.g. 2026-10-26_16-03-45
STAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOG="$LOG_DIR/${NAME}_${STAMP}.log"
PID_FILE="$LOG_DIR/${NAME}_${STAMP}.pid"

# Header inside the log, so the file itself says when and how it started.
{
    echo "=== Run started: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
    echo "=== Command: mojo $TARGET $* ==="
    echo "=== Host: $(uname -srm) ==="
    echo
} > "$LOG"

# Detach from the terminal. setsid is not available on macOS, so fall back
# to plain nohup there.
if command -v setsid >/dev/null 2>&1; then
    setsid nohup mojo "$TARGET" "$@" >> "$LOG" 2>&1 < /dev/null &
else
    nohup mojo "$TARGET" "$@" >> "$LOG" 2>&1 < /dev/null &
fi
echo $! > "$PID_FILE"

# Convenience pointer to the newest log; ignore failure (e.g. no symlinks).
ln -sf "${NAME}_${STAMP}.log" "$LOG_DIR/${NAME}_latest.log" 2>/dev/null || true

echo "Started $TARGET (pid $(cat "$PID_FILE"))"
echo "  log: $SCRIPT_DIR/$LOG"
echo "  pid: $SCRIPT_DIR/$PID_FILE"
echo "Follow with: tail -f $LOG_DIR/${NAME}_latest.log"
echo "Stop with:   kill \$(cat $PID_FILE)"
