#!/usr/bin/env bash
# setup.sh — bootstrap the consciousness simulation environment
set -euo pipefail

PYTHON=${PYTHON:-python3}
OLLAMA_API="http://localhost:11434"
MODEL_LARGE="llama3.1"    # requires ~4.8 GiB RAM
MODEL_SMALL="llama3.2:3b" # requires ~2.0 GiB RAM
MEM_THRESHOLD_GB=4        # use large model only if more than this is free

info()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[  ok]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()   { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. System deps
# ---------------------------------------------------------------------------
info "Checking system dependencies..."

if ! command -v zstd &>/dev/null; then
    info "Installing zstd..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y zstd
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y zstd
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm zstd
    elif command -v brew &>/dev/null; then
        brew install zstd
    else
        die "Cannot install zstd — please install it manually and rerun."
    fi
fi
ok "zstd present"

# ---------------------------------------------------------------------------
# 2. Ollama
# ---------------------------------------------------------------------------
info "Checking Ollama..."

if ! command -v ollama &>/dev/null; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
else
    ok "Ollama already installed ($(ollama --version 2>/dev/null || echo 'version unknown'))"
fi

# ---------------------------------------------------------------------------
# 3. Start Ollama server if not running
# ---------------------------------------------------------------------------
if ! curl -sf "${OLLAMA_API}/api/tags" &>/dev/null; then
    info "Starting Ollama server..."
    ollama serve >/tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    # Wait up to 10 s for the API to become available.
    for i in $(seq 1 10); do
        sleep 1
        if curl -sf "${OLLAMA_API}/api/tags" &>/dev/null; then
            ok "Ollama server is up (pid $OLLAMA_PID)"
            break
        fi
        if [ "$i" -eq 10 ]; then
            die "Ollama server did not start. Check /tmp/ollama.log for details."
        fi
    done
else
    ok "Ollama server already running"
fi

# ---------------------------------------------------------------------------
# 4. Pick and pull model based on available memory
# ---------------------------------------------------------------------------
info "Checking available memory..."

free_gb=0
if command -v free &>/dev/null; then
    free_gb=$(free -g | awk '/^Mem:/ { print $7 }')
elif [[ "$(uname)" == "Darwin" ]]; then
    # macOS: approximate from vm_stat
    pages_free=$(vm_stat | awk '/Pages free/ { gsub(/\./, "", $3); print $3 }')
    free_gb=$(( pages_free * 4096 / 1024 / 1024 / 1024 ))
fi

if [ "${free_gb:-0}" -ge "$MEM_THRESHOLD_GB" ] 2>/dev/null; then
    MODEL="$MODEL_LARGE"
else
    MODEL="$MODEL_SMALL"
    warn "Only ~${free_gb} GiB free — using ${MODEL_SMALL} instead of ${MODEL_LARGE}."
    warn "To use the larger model, free up memory and rerun with: PYTHON=$PYTHON $0"
fi

# Check if model is already pulled.
if curl -sf "${OLLAMA_API}/api/tags" | grep -q "\"${MODEL}\""; then
    ok "Model '${MODEL}' already available"
else
    info "Pulling model '${MODEL}' (this may take a few minutes)..."
    ollama pull "${MODEL}"
    ok "Model '${MODEL}' ready"
fi

# ---------------------------------------------------------------------------
# 5. Update default config to match pulled model
# ---------------------------------------------------------------------------
CONFIG="consciousness-sim/config/default_consciousness.yaml"
if [ -f "$CONFIG" ]; then
    # Only patch if the current model differs from what's configured.
    current=$(grep -m1 '^\s*model:' "$CONFIG" | awk '{print $2}' | tr -d '"')
    if [ "$current" != "$MODEL" ]; then
        info "Updating config model from '${current}' to '${MODEL}'..."
        sed -i "s|model: \"${current}\"|model: \"${MODEL}\"|" "$CONFIG"
        ok "Config updated"
    fi
fi

# ---------------------------------------------------------------------------
# 6. Python dependencies
# ---------------------------------------------------------------------------
info "Installing Python dependencies..."
cd consciousness-sim
"$PYTHON" -m pip install -r requirements.txt --quiet
"$PYTHON" -m pip install -r requirements-dev.txt --quiet
ok "Python dependencies installed"
cd ..

# ---------------------------------------------------------------------------
# 7. Smoke test
# ---------------------------------------------------------------------------
info "Running smoke test against Ollama API..."
response=$(curl -sf "${OLLAMA_API}/api/generate" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"Reply with one word: ready\",\"stream\":false}" \
    | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin).get('response','').strip())" 2>/dev/null || echo "")

if [ -n "$response" ]; then
    ok "Ollama responded: \"${response}\""
else
    warn "Smoke test got no response — model may be loading. Try running the sim anyway."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
printf '\n'
ok "Setup complete. Run the simulation with:"
printf '    cd consciousness-sim && %s scripts/spawn.py --name "Aria"\n' "$PYTHON"
