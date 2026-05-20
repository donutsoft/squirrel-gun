#!/usr/bin/env bash
set -euo pipefail

duration="${1:?Usage: ValveController.sh <duration_seconds>}"
chip="${GPIO_CHIP:-gpiochip4}"
line="${VALVE_GPIO_LINE:-24}"
lock="/tmp/valve_controller.lock"

off() {
  gpioset "$chip" "$line=0" || true
}

(
  flock -n 9 || {
    echo "ValveController already running; exiting."
    exit 0
  }

  trap off EXIT INT TERM HUP

  gpioset "$chip" "$line=1"
  sleep "$duration"
  off
) 9>"$lock"
