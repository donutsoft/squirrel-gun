import os

from .MqttClient import MqttClient
from .SerialHardwareController import get_serial_hardware_controller


class WaterController:
    def __init__(self):
        self.mqtt = MqttClient()
        self._pin = int(os.environ.get("SQUIRREL_VALVE_PIN", "2"))
        self._serial = get_serial_hardware_controller()

    def startWatering(self, duration):
        self._serial.trace_event(
            f"water-start duration_sec={duration!r} pin={self._pin}"
        )
        print(f"Starting watering for {duration} seconds.")
        duration_ms = max(0, int(round(float(duration) * 1000.0)))
        response = self._serial.command("TIMED-ON", self._pin, duration_ms)
        self._serial.trace_event(
            f"water-serial-complete duration_ms={duration_ms} pin={self._pin} response={response!r}"
        )

        self.mqtt.publish("squirrel/fire", '{"state": "fired"}')
        self._serial.trace_event("water-mqtt-complete")
