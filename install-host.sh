#!/usr/bin/env sh
# Install vllm-rdna-dash directly on the host (no container). Most people want install.sh (Docker).
#
#   curl -fsSL https://raw.githubusercontent.com/crispyging/vllm-rdna-dash/main/install-host.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/crispyging/vllm-rdna-dash/main/install-host.sh | sh -s -- --start      # install and start it
#
# Puts the tool in ~/.local/share/vllm-rdna-dash and a `rdna-dash` command in
# ~/.local/bin. Needs only git and Python 3.8+. Never uses sudo.
set -eu

REPO_URL="${RDNA_DASH_REPO:-https://github.com/crispyging/vllm-rdna-dash.git}"
DEST="${RDNA_DASH_HOME:-$HOME/.local/share/vllm-rdna-dash}"
BIN="$HOME/.local/bin"
START=0
for a in "$@"; do [ "$a" = "--start" ] && START=1; done

say() { printf '\033[1m%s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1; }

PY=""
for p in python3 python3.12 python3.11 python3.10 python3.9 python3.8; do
    if need "$p" && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then PY="$p"; break; fi
done
if [ -z "$PY" ]; then
    echo "Python 3.8 or newer is needed. Install it with your package manager:"
    echo "  apt: sudo apt install python3    dnf: sudo dnf install python3"
    echo "  zypper: sudo zypper install python3    pacman: sudo pacman -S python"
    exit 1
fi
if ! need git; then
    echo "git is needed. Install it with your package manager (apt/dnf/zypper/pacman install git)."
    exit 1
fi

if [ -d "$DEST/.git" ]; then
    say "Updating $DEST"
    git -C "$DEST" pull --ff-only
else
    say "Installing to $DEST"
    mkdir -p "$(dirname "$DEST")"
    git clone --depth 1 "$REPO_URL" "$DEST"
fi

mkdir -p "$BIN"
cat > "$BIN/rdna-dash" <<EOF
#!/usr/bin/env sh
cd "$DEST" && exec "$PY" -m rdna_dash "\$@"
EOF
chmod +x "$BIN/rdna-dash"

case ":$PATH:" in *":$BIN:"*) ;; *) say "Add $BIN to your PATH (e.g. in ~/.bashrc): export PATH=\"$BIN:\$PATH\"" ;; esac

say "Done."
echo "  rdna-dash doctor          check this box and list what is missing"
echo "  rdna-dash                 open the dashboard on http://127.0.0.1:7860"
echo "  rdna-dash --host 0.0.0.0  reach it from another machine (prints a token link)"

if [ "$START" = 1 ]; then exec "$BIN/rdna-dash"; fi
