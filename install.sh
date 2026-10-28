#!/usr/bin/env sh
# Install (or update) the vllm-rdna-dash dashboard as a container:
#
#   curl -fsSL https://raw.githubusercontent.com/crispyging/vllm-rdna-dash/main/install.sh | sh
#
# Works with Docker or Podman (rootless too). Everything lives in ~/vllm-rdna
# (set DATA=/some/path to change it). Run it again to update.
set -eu

RAW="https://raw.githubusercontent.com/crispyging/vllm-rdna-dash/main"
DATA="${DATA:-$HOME/vllm-rdna}"
PORT="${PORT:-7860}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---- find a container engine this user can drive ------------------------------
ENGINE=""; SUDO=""; SOCK=""; FLAVOR="docker"
if have docker && docker info >/dev/null 2>&1; then
    ENGINE=docker
elif have podman && podman info >/dev/null 2>&1; then
    ENGINE=podman
elif have docker && have sudo; then
    say "Your user cannot reach Docker; using sudo for the install (add yourself to the docker group to skip this)."
    sudo docker info >/dev/null 2>&1 || die "Docker is installed but not running. Start it: sudo systemctl enable --now docker"
    ENGINE=docker; SUDO=sudo
else
    die "No Docker or Podman found. Install one first:
  Ubuntu/Debian: sudo apt install docker.io docker-compose-v2
  Fedora/RHEL:   sudo dnf install podman podman-compose
  Arch:          sudo pacman -S docker docker-compose
  openSUSE:      sudo zypper install docker docker-compose"
fi

if [ "$ENGINE" = docker ]; then
    SOCK=/var/run/docker.sock
    if $SUDO docker compose version >/dev/null 2>&1; then COMPOSE="$SUDO docker compose"
    elif have docker-compose; then COMPOSE="$SUDO docker-compose"
    else die "Docker Compose is missing. Ubuntu/Debian: sudo apt install docker-compose-v2"; fi
else
    FLAVOR=podman
    if [ "$(podman info --format '{{.Host.Security.Rootless}}' 2>/dev/null)" = true ]; then
        FLAVOR=podman-rootless
        SOCK="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/podman/podman.sock"
        have systemctl && systemctl --user enable --now podman.socket >/dev/null 2>&1 || true
    else
        SOCK=/run/podman/podman.sock
    fi
    [ -S "$SOCK" ] || die "Podman's API socket is not running at $SOCK.
  Rootless: systemctl --user enable --now podman.socket
  Rootful:  sudo systemctl enable --now podman.socket"
    if podman compose version >/dev/null 2>&1; then COMPOSE="podman compose"
    elif have podman-compose; then COMPOSE="podman-compose"
    else die "No compose for Podman. Install podman-compose (dnf/apt/zypper install podman-compose)."; fi
fi

# ---- GPU sanity (warn only: the dashboard explains what is missing) ----------------
[ -e /dev/kfd ] || say "Note: /dev/kfd is missing, so ROCm is not set up yet. The dashboard will show what to install."

# ---- files ----------------------------------------------------------------------
mkdir -p "$DATA"
cd "$DATA"
say "Fetching compose.yml into $DATA"
curl -fsSL "$RAW/compose.yml" -o compose.yml
if [ "$FLAVOR" = podman-rootless ]; then
    sed -i 's/^    group_add: \[video, render\].*/    group_add: [keep-groups]/' compose.yml
fi

if [ -f .env ]; then
    say "Keeping your existing $DATA/.env"
else
    TOKEN="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    cat > .env <<EOF
# vllm-rdna-dash settings. Edit, then run: $COMPOSE up -d
DATA=$DATA
PORT=$PORT
RDNA_DASH_TOKEN=$TOKEN
BOX_NAME=$(hostname 2>/dev/null || echo gpu-box)
ENGINE=$FLAVOR
DOCKER_SOCK=$SOCK
# HF_TOKEN=   # only for gated models
EOF
    chmod 600 .env
fi

# ---- start ------------------------------------------------------------------------
say "Pulling and starting the dashboard"
$COMPOSE pull
$COMPOSE up -d

i=0
until curl -s -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; do
    i=$((i + 1)); [ "$i" -gt 60 ] && die "The dashboard did not come up. Check: $COMPOSE logs"
    sleep 1
done

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ -n "$IP" ] || IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')"
TOKEN="$(sed -n 's/^RDNA_DASH_TOKEN=//p' .env)"

echo
say "vllm-rdna-dash is running."
echo "  Open:   http://${IP:-<this-box>}:$PORT   (from any machine on your network)"
echo "  Token:  $TOKEN"
echo "  Files:  $DATA   (settings in $DATA/.env)"
echo "  Update: run this installer again.   Stop: cd $DATA && $COMPOSE down"
