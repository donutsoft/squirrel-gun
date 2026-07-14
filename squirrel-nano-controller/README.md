# Squirrel Nano Controller

Arduino Nano firmware for the squirrel hardware controller.

Serial settings:

- Port on the Raspberry Pi: `/dev/ttyUSB0`
- Baud: `115200`
- Commands are newline-terminated.
- Successful commands respond with `OK`.

Supported commands:

```text
PANTILT <pan-pin> <tilt-pin> <pan-degrees> <tilt-degrees>
TIMED-ON <pin> <duration-ms>
ON <pin>
OFF <pin>
```

`TIMED-ON` is non-blocking: the Nano immediately returns `OK`, then turns the pin
off from its own `millis()` loop after the requested duration.

On boot, firmware drives every controllable pin D2-D13 and A0-A5 low before
accepting serial commands. D0/D1 are left alone for USB serial.

The daemon defaults to these Nano pins:

- Pan servo: D5 (`SQUIRREL_PAN_PIN`)
- Tilt servo: D4 (`SQUIRREL_TILT_PIN`)
- Laser: D3 (`SQUIRREL_LASER_PIN`)
- Valve: D2 (`SQUIRREL_VALVE_PIN`)

Override those environment variables if you wire the Nano differently.

Build/upload with PlatformIO:

```sh
pio run -d squirrel-nano-controller -t upload --upload-port /dev/ttyUSB0
```
