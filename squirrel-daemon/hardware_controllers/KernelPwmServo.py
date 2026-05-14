import os
import time
from pathlib import Path
from typing import Union


class KernelPwmServo:
    PERIOD_NS = 20_000_000

    def __init__(
        self,
        channel: int,
        *,
        chip: str = "pwmchip0",
        min_pulse_us: int = 500,
        max_pulse_us: int = 2500,
        min_angle: float = 0.0,
        max_angle: float = 180.0,
    ):
        self.channel = int(channel)
        self.chip_path = Path("/sys/class/pwm") / chip
        self.pwm_path = self.chip_path / f"pwm{self.channel}"
        self.min_pulse_us = int(min_pulse_us)
        self.max_pulse_us = int(max_pulse_us)
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        self._available = self._setup()

    def _write(self, name: str, value: Union[int, str]) -> None:
        (self.pwm_path / name).write_text(f"{value}\n", encoding="ascii")

    def _setup(self) -> bool:
        if not self.chip_path.exists():
            print(
                f"[KernelPwmServo] {self.chip_path} not found; "
                "enable the pwm-2chan overlay for GPIO 18/19."
            )
            return False

        if not self.pwm_path.exists():
            try:
                (self.chip_path / "export").write_text(f"{self.channel}\n", encoding="ascii")
            except OSError as exc:
                if not self.pwm_path.exists():
                    print(f"[KernelPwmServo] Could not export PWM channel {self.channel}: {exc}")
                    return False

        for _ in range(20):
            if self.pwm_path.exists():
                break
            time.sleep(0.01)

        try:
            self._write("enable", 0)
        except OSError:
            pass

        try:
            self._write("period", self.PERIOD_NS)
            self._write("duty_cycle", 0)
            self._write("enable", 1)
        except OSError as exc:
            print(f"[KernelPwmServo] Could not initialize {self.pwm_path}: {exc}")
            return False

        return True

    def set_angle(self, angle: float) -> None:
        if not self._available:
            return

        clamped = max(self.min_angle, min(self.max_angle, float(angle)))
        span = self.max_angle - self.min_angle
        fraction = 0.0 if span == 0 else (clamped - self.min_angle) / span
        pulse_us = self.min_pulse_us + fraction * (self.max_pulse_us - self.min_pulse_us)
        self._write("duty_cycle", int(round(pulse_us * 1000)))
