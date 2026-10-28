#!/usr/bin/env bash
# Set a power cap on Radeon PRO V620 cards (stock is 250 W; the community runs 180 W).
#
#   sudo ./tools/power.sh status
#   sudo ./tools/power.sh 180              # apply now (until reboot)
#   sudo ./tools/power.sh 180 --persist    # apply now and at every boot (systemd unit)
#   sudo ./tools/power.sh --remove         # drop the boot unit (cards return to stock on reboot)
#
# Many kernels reject caps below the 250 W VBIOS floor with "Invalid argument" until
# the v620_toolbox powerfix is applied: https://blivioniag.github.io/gfx1030-wiki/tuning/power.html
set -euo pipefail

UNIT=/etc/systemd/system/v620-power-cap.service
INSTALLED=/usr/local/sbin/v620-power-cap

hwmons() {
    for d in /sys/class/drm/card[0-9]*/device; do
        [ "$(cat "$d/vendor" 2>/dev/null)" = "0x1002" ] || continue
        [ "$(cat "$d/device" 2>/dev/null)" = "0x73a1" ] || continue
        for h in "$d"/hwmon/hwmon*; do [ -e "$h/power1_cap" ] && echo "$h"; done
    done | sort -u
}

w() { awk -v u="$1" 'BEGIN { printf "%d", u / 1000000 }'; }

status() {
    local found=0
    for h in $(hwmons); do
        found=1
        echo "$(basename "$(readlink -f "$h/../..")")  cap $(w "$(cat "$h/power1_cap")") W  (min $(w "$(cat "$h/power1_cap_min" 2>/dev/null || echo 0)") W, max $(w "$(cat "$h/power1_cap_max" 2>/dev/null || echo 0)") W)  now $(w "$(cat "$h/power1_average" 2>/dev/null || cat "$h/power1_input" 2>/dev/null || echo 0)") W"
    done
    [ "$found" = 1 ] || echo "No Radeon PRO V620 found."
    [ -f "$UNIT" ] && echo "Boot unit: installed ($UNIT)" || echo "Boot unit: not installed"
}

apply() {
    local watts="$1" ok=0 bad=0
    for h in $(hwmons); do
        if echo "$((watts * 1000000))" > "$h/power1_cap" 2>/dev/null; then
            ok=$((ok + 1))
        else
            bad=$((bad + 1))
        fi
    done
    echo "Set ${watts} W on $ok card(s)."
    if [ "$bad" -gt 0 ]; then
        echo "$bad card(s) refused ${watts} W. This kernel enforces the 250 W VBIOS floor." >&2
        echo "Apply the v620_toolbox powerfix, then run this again:" >&2
        echo "  https://github.com/blivioniag/v620_toolbox (powertuning/ or ubuntu_powertuning/)" >&2
        return 1
    fi
}

need_root() { [ "$(id -u)" = 0 ] || { echo "Run with sudo." >&2; exit 1; }; }

case "${1:-status}" in
    status) status ;;
    --remove)
        need_root
        systemctl disable --now v620-power-cap.service 2>/dev/null || true
        rm -f "$UNIT" "$INSTALLED"
        systemctl daemon-reload
        echo "Removed the boot unit. Cards go back to stock power on the next reboot."
        ;;
    *[!0-9]*) echo "usage: sudo $0 status | <watts> [--persist] | --remove" >&2; exit 2 ;;
    *)
        need_root
        watts="$1"
        if [ "$watts" -lt 120 ] || [ "$watts" -gt 250 ]; then echo "Pick a cap between 120 and 250 W." >&2; exit 2; fi
        apply "$watts"
        if [ "${2:-}" = "--persist" ]; then
            if [ -f "$0" ] && grep -q "v620-power-cap" "$0" 2>/dev/null; then
                install -m 0755 "$0" "$INSTALLED"
            else  # run as `curl ... | sudo bash`: fetch a copy for the boot unit
                curl -fsSL https://raw.githubusercontent.com/crispyging/vllm-rdna-dash/main/tools/power.sh -o "$INSTALLED"
                chmod 0755 "$INSTALLED"
            fi
            cat > "$UNIT" <<EOF
[Unit]
Description=Cap Radeon PRO V620 power at ${watts} W
After=systemd-modules-load.service

[Service]
Type=oneshot
ExecStart=$INSTALLED $watts

[Install]
WantedBy=multi-user.target
EOF
            systemctl daemon-reload
            systemctl enable v620-power-cap.service >/dev/null
            echo "Installed $UNIT: ${watts} W is applied at every boot."
        fi
        status
        ;;
esac
