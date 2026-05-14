#!/usr/bin/env bash
set -e

PWM_CHIP="${SQUIRREL_PWM_CHIP:-pwmchip0}"
PAN_PWM_CHANNEL="${SQUIRREL_PAN_PWM_CHANNEL:-2}"
TILT_PWM_CHANNEL="${SQUIRREL_TILT_PWM_CHANNEL:-3}"
PWM_ROOT="/sys/class/pwm/${PWM_CHIP}"

prepare_pwm_channel() {
    local channel="$1"
    local channel_path="${PWM_ROOT}/pwm${channel}"

    if [ ! -d "${PWM_ROOT}" ]; then
        echo "PWM chip ${PWM_ROOT} is missing; enable dtoverlay=pwm-2chan and reboot." >&2
        return 1
    fi

    if [ ! -d "${channel_path}" ]; then
        echo "${channel}" | sudo tee "${PWM_ROOT}/export" >/dev/null
    fi

    for _ in 1 2 3 4 5 6 7 8 9 10; do
        [ -d "${channel_path}" ] && break
        sleep 0.1
    done

    sudo chown -R "$(id -u):$(id -g)" "${channel_path}"
    chmod u+rw "${channel_path}/period" "${channel_path}/duty_cycle" "${channel_path}/enable"
}

prepare_pwm_channel "${PAN_PWM_CHANNEL}"
prepare_pwm_channel "${TILT_PWM_CHANNEL}"

# Enable threaded dev server so long-lived stream doesn't block other requests
uv run python -m flask run --host="0.0.0.0" --with-threads &
