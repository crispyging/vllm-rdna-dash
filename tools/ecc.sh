#!/usr/bin/env bash
# Turn VRAM ECC off (or back on) on Radeon PRO V620 / W6800 cards.
#
#   sudo ./tools/ecc.sh status   # what the cards and kernel command line say now
#   sudo ./tools/ecc.sh off      # add amdgpu.ras_enable=0, then reboot TWICE
#   sudo ./tools/ecc.sh on       # remove it, then reboot twice
#
# The firmware arms the change on the first boot with the parameter and applies it
# on the second. Method: https://blivioniag.github.io/gfx1030-wiki/tuning/ecc.html
# Leave ECC on if any card already reports ECC errors.
set -euo pipefail

PARAM="amdgpu.ras_enable=0"
MODE="${1:-status}"

pro_cards() {
    for d in /sys/class/drm/card[0-9]*/device; do
        [ "$(cat "$d/vendor" 2>/dev/null)" = "0x1002" ] || continue
        case "$(cat "$d/device" 2>/dev/null)" in 0x73a1|0x73a3|0x73ab) echo "$d" ;; esac
    done | sort -u
}

status() {
    local found=0 errors=0
    echo "Kernel command line: $(grep -q "$PARAM" /proc/cmdline && echo "has $PARAM" || echo "no $PARAM")"
    for d in $(pro_cards); do
        found=1
        local total gb counts
        total=$(cat "$d/mem_info_vram_total" 2>/dev/null || echo 0)
        gb=$(awk -v b="$total" 'BEGIN { printf "%.1f", b / 1073741824 }')
        counts=$(tr '\n' ' ' < "$d/ras/umc_err_count" 2>/dev/null || echo "n/a")
        echo "$(basename "$(readlink -f "$d")")  VRAM ${gb} GB  ECC errors: ${counts}"
        case "$counts" in *"ce: 0"*"ue: 0"*|*"ue: 0"*"ce: 0"*|n/a) ;; *) errors=1 ;; esac
    done
    [ "$found" = 1 ] || { echo "No Radeon PRO V620 / W6800 found. Consumer cards have no ECC."; return 0; }
    dmesg 2>/dev/null | grep -iE "GECC|MEM ECC" | tail -n 4 || true
    [ "$errors" = 0 ] || echo "WARNING: a card reports ECC errors. Keep ECC on and stress-test that card."
    return 0
}

need_root() { [ "$(id -u)" = 0 ] || { echo "Run with sudo: sudo $0 $MODE" >&2; exit 1; }; }

edit_default_grub() {
    # $1 = add | remove
    local f=/etc/default/grub
    [ -f "$f" ] || return 1
    cp -a "$f" "$f.bak-vllm-rdna-$(date +%Y%m%d%H%M%S)"
    if [ "$1" = add ]; then
        grep -q "$PARAM" "$f" || sed -i -E "s/^(GRUB_CMDLINE_LINUX_DEFAULT=\")/\1$PARAM /" "$f"
    else
        sed -i -E "s/ ?$PARAM//g" "$f"
    fi
    if command -v update-grub >/dev/null; then update-grub
    elif command -v grub2-mkconfig >/dev/null; then grub2-mkconfig -o "$( [ -d /boot/grub2 ] && echo /boot/grub2/grub.cfg || echo /boot/grub/grub.cfg)"
    elif command -v grub-mkconfig >/dev/null; then grub-mkconfig -o /boot/grub/grub.cfg
    else echo "Edited $f but found no grub config generator; regenerate it yourself." >&2
    fi
}

apply() {
    # $1 = add | remove
    if command -v grubby >/dev/null; then            # Fedora, RHEL, Rocky, Alma
        if [ "$1" = add ]; then grubby --update-kernel=ALL --args="$PARAM"
        else grubby --update-kernel=ALL --remove-args="$PARAM"; fi
    elif command -v kernelstub >/dev/null; then      # Pop!_OS
        if [ "$1" = add ]; then kernelstub -a "$PARAM"; else kernelstub -d "$PARAM"; fi
    elif [ -f /etc/kernel/cmdline ] && ! [ -f /etc/default/grub ]; then   # systemd-boot + kernel-install
        cp -a /etc/kernel/cmdline "/etc/kernel/cmdline.bak-vllm-rdna-$(date +%Y%m%d%H%M%S)"
        if [ "$1" = add ]; then grep -q "$PARAM" /etc/kernel/cmdline || sed -i "s/$/ $PARAM/" /etc/kernel/cmdline
        else sed -i -E "s/ ?$PARAM//g" /etc/kernel/cmdline; fi
        echo "Updated /etc/kernel/cmdline. Rebuild boot entries (e.g. 'kernel-install add' or your distro's tool)."
    elif edit_default_grub "$1"; then :
    else
        echo "Could not find grubby, kernelstub, systemd-boot or /etc/default/grub." >&2
        echo "Add or remove '$PARAM' on the kernel command line by hand." >&2
        exit 1
    fi
}

case "$MODE" in
    status) status ;;
    off)
        need_root
        if status | grep -q "WARNING"; then
            echo "Refusing: a card reports ECC errors. Fix or replace it first." >&2
            exit 1
        fi
        apply add
        echo
        echo "Done. Now reboot TWICE: the first boot arms it, the second frees the ~2 GB per card."
        echo "After the second boot, 'sudo $0 status' should show about 32 GB per card."
        ;;
    on)
        need_root
        apply remove
        echo
        echo "Done. Reboot twice to turn ECC back on."
        ;;
    *) echo "usage: sudo $0 status|off|on" >&2; exit 2 ;;
esac
