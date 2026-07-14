#!/usr/bin/env python3
"""Manual test console for the Arduino Nano hardware controller."""

import argparse
import shlex
import sys
import time
from typing import Iterable, List, Optional

from hardware_controllers.SerialHardwareController import SerialHardwareController


FIRST_PIN = 2
LAST_PIN = 19
DEFAULT_PULSE_MS = 500
DEFAULT_STEP_DELAY_SEC = 0.7


class HardwareTester:
    def __init__(self, controller: SerialHardwareController):
        self.controller = controller

    def send(self, line: str) -> None:
        print(f"> {line}")
        response = self.controller.send_line(line)
        print(response)

    def off_all(self, start: int = FIRST_PIN, end: int = LAST_PIN) -> None:
        for pin in range(start, end + 1):
            self.send(f"OFF {pin}")

    def pulse(self, pin: int, duration_ms: int = DEFAULT_PULSE_MS) -> None:
        self.send(f"TIMED-ON {pin} {duration_ms}")

    def scan_outputs(
        self,
        pins: Iterable[int],
        duration_ms: int = DEFAULT_PULSE_MS,
        step_delay_sec: float = DEFAULT_STEP_DELAY_SEC,
        wait_for_enter: bool = True,
    ) -> None:
        for pin in pins:
            if wait_for_enter:
                input(f"Press Enter to pulse pin {pin} for {duration_ms} ms...")
            self.pulse(pin, duration_ms)
            time.sleep(max(0.0, step_delay_sec))

    def pantilt(self, pan_pin: int, tilt_pin: int, pan: float, tilt: float) -> None:
        self.send(f"PANTILT {pan_pin} {tilt_pin} {pan:g} {tilt:g}")

    def servo_wiggle(
        self,
        pan_pin: int,
        tilt_pin: int,
        center_pan: float = 135.0,
        center_tilt: float = 90.0,
        delta: float = 25.0,
        step_delay_sec: float = 0.8,
    ) -> None:
        points = [
            (center_pan, center_tilt),
            (center_pan - delta, center_tilt),
            (center_pan + delta, center_tilt),
            (center_pan, center_tilt),
            (center_pan, center_tilt - delta),
            (center_pan, center_tilt + delta),
            (center_pan, center_tilt),
        ]
        for pan, tilt in points:
            self.pantilt(pan_pin, tilt_pin, pan, tilt)
            time.sleep(max(0.0, step_delay_sec))


def parse_pin(text: str) -> int:
    pin = int(text)
    if pin < FIRST_PIN or pin > LAST_PIN:
        raise ValueError(f"pin must be {FIRST_PIN}-{LAST_PIN}")
    return pin


def parse_pin_range(args: List[str]) -> range:
    if not args:
        return range(FIRST_PIN, LAST_PIN + 1)
    if len(args) == 1:
        pin = parse_pin(args[0])
        return range(pin, pin + 1)
    if len(args) == 2:
        start = parse_pin(args[0])
        end = parse_pin(args[1])
        if end < start:
            raise ValueError("end pin must be >= start pin")
        return range(start, end + 1)
    raise ValueError("expected: [start-pin [end-pin]]")


HELP = """Commands:
  raw <command line>                 Send an exact command, e.g. raw OFF 2
  on <pin>                           Latch a pin HIGH
  off <pin>                          Latch a pin LOW
  pulse <pin> [duration-ms]          Turn a pin on briefly, then MCU turns it off
  scan [start-pin [end-pin]] [ms]    Pulse pins one at a time after Enter prompts
  off-all [start-pin [end-pin]]      Send OFF to each pin in the range
  pantilt <pan-pin> <tilt-pin> <pan-deg> <tilt-deg>
  wiggle <pan-pin> <tilt-pin> [delta-deg]
  help
  quit

Pins are Nano D2-D13 and A0-A5, represented as 2-19. D0/D1 are reserved for serial.
Use pulse/scan first for valves, relays, and lasers. Use on only when you mean it.
"""


def run_repl(tester: HardwareTester) -> int:
    print(HELP)
    while True:
        try:
            text = input("nano> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue

        try:
            parts = shlex.split(text)
            command = parts[0].lower()
            args = parts[1:]

            if command in ("quit", "exit"):
                return 0
            if command == "help":
                print(HELP)
            elif command == "raw":
                if not args:
                    raise ValueError("usage: raw <command line>")
                tester.send(" ".join(args))
            elif command == "on":
                if len(args) != 1:
                    raise ValueError("usage: on <pin>")
                tester.send(f"ON {parse_pin(args[0])}")
            elif command == "off":
                if len(args) != 1:
                    raise ValueError("usage: off <pin>")
                tester.send(f"OFF {parse_pin(args[0])}")
            elif command == "pulse":
                if len(args) not in (1, 2):
                    raise ValueError("usage: pulse <pin> [duration-ms]")
                pin = parse_pin(args[0])
                duration_ms = int(args[1]) if len(args) == 2 else DEFAULT_PULSE_MS
                tester.pulse(pin, duration_ms)
            elif command == "scan":
                duration_ms = DEFAULT_PULSE_MS
                pin_args = args
                if len(args) in (2, 3) and args[-1].lower().endswith("ms"):
                    duration_ms = int(args[-1][:-2])
                    pin_args = args[:-1]
                elif len(args) == 3:
                    duration_ms = int(args[-1])
                    pin_args = args[:-1]
                tester.scan_outputs(parse_pin_range(pin_args), duration_ms)
            elif command == "off-all":
                pins = parse_pin_range(args)
                tester.off_all(pins.start, pins.stop - 1)
            elif command == "pantilt":
                if len(args) != 4:
                    raise ValueError("usage: pantilt <pan-pin> <tilt-pin> <pan-deg> <tilt-deg>")
                tester.pantilt(parse_pin(args[0]), parse_pin(args[1]), float(args[2]), float(args[3]))
            elif command == "wiggle":
                if len(args) not in (2, 3):
                    raise ValueError("usage: wiggle <pan-pin> <tilt-pin> [delta-deg]")
                delta = float(args[2]) if len(args) == 3 else 25.0
                tester.servo_wiggle(parse_pin(args[0]), parse_pin(args[1]), delta=delta)
            else:
                raise ValueError(f"unknown command: {command}")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send manual test commands to the squirrel Nano controller."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="serial port")
    parser.add_argument("--baud", type=int, default=115200, help="serial baud")
    parser.add_argument("--timeout", type=float, default=2.0, help="response timeout seconds")
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
        help="delay after opening serial, for Nano auto-reset",
    )

    subcommands = parser.add_subparsers(dest="command")

    raw = subcommands.add_parser("raw", help="send an exact command line")
    raw.add_argument("line", nargs=argparse.REMAINDER)

    pulse = subcommands.add_parser("pulse", help="send TIMED-ON")
    pulse.add_argument("pin", type=parse_pin)
    pulse.add_argument("duration_ms", type=int, nargs="?", default=DEFAULT_PULSE_MS)

    on = subcommands.add_parser("on", help="send ON")
    on.add_argument("pin", type=parse_pin)

    off = subcommands.add_parser("off", help="send OFF")
    off.add_argument("pin", type=parse_pin)

    off_all = subcommands.add_parser("off-all", help="send OFF to a range of pins")
    off_all.add_argument("start_pin", type=parse_pin, nargs="?", default=FIRST_PIN)
    off_all.add_argument("end_pin", type=parse_pin, nargs="?", default=LAST_PIN)

    scan = subcommands.add_parser("scan", help="pulse pins one at a time")
    scan.add_argument("start_pin", type=parse_pin, nargs="?", default=FIRST_PIN)
    scan.add_argument("end_pin", type=parse_pin, nargs="?", default=LAST_PIN)
    scan.add_argument("--duration-ms", type=int, default=DEFAULT_PULSE_MS)
    scan.add_argument("--no-prompt", action="store_true")

    pantilt = subcommands.add_parser("pantilt", help="send PANTILT")
    pantilt.add_argument("pan_pin", type=parse_pin)
    pantilt.add_argument("tilt_pin", type=parse_pin)
    pantilt.add_argument("pan_degrees", type=float)
    pantilt.add_argument("tilt_degrees", type=float)

    wiggle = subcommands.add_parser("wiggle", help="move a pan/tilt pair around center")
    wiggle.add_argument("pan_pin", type=parse_pin)
    wiggle.add_argument("tilt_pin", type=parse_pin)
    wiggle.add_argument("--delta", type=float, default=25.0)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    controller = SerialHardwareController(
        port=args.port,
        baud=args.baud,
        timeout_sec=args.timeout,
    )
    controller.startup_delay_sec = args.startup_delay
    tester = HardwareTester(controller)

    try:
        if args.command is None:
            return run_repl(tester)
        if args.command == "raw":
            if not args.line:
                parser.error("raw requires a command line")
            tester.send(" ".join(args.line))
        elif args.command == "pulse":
            tester.pulse(args.pin, args.duration_ms)
        elif args.command == "on":
            tester.send(f"ON {args.pin}")
        elif args.command == "off":
            tester.send(f"OFF {args.pin}")
        elif args.command == "off-all":
            if args.end_pin < args.start_pin:
                parser.error("end_pin must be >= start_pin")
            tester.off_all(args.start_pin, args.end_pin)
        elif args.command == "scan":
            if args.end_pin < args.start_pin:
                parser.error("end_pin must be >= start_pin")
            tester.scan_outputs(
                range(args.start_pin, args.end_pin + 1),
                args.duration_ms,
                wait_for_enter=not args.no_prompt,
            )
        elif args.command == "pantilt":
            tester.pantilt(args.pan_pin, args.tilt_pin, args.pan_degrees, args.tilt_degrees)
        elif args.command == "wiggle":
            tester.servo_wiggle(args.pan_pin, args.tilt_pin, delta=args.delta)
        return 0
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
