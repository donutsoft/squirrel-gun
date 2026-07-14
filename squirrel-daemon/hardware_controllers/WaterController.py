import os

from .MqttClient import MqttClient
from .SerialHardwareController import get_serial_hardware_controller


class WaterController:
    def __init__(self):
        self.mqtt = MqttClient()
        self._pin = int(os.environ.get("SQUIRREL_VALVE_PIN", "2"))
        self._serial = get_serial_hardware_controller()

    def startWatering(self, duration):
        print(f"Starting watering for {duration} seconds.")
        duration_ms = max(0, int(round(float(duration) * 1000.0)))
        self._serial.command("TIMED-ON", self._pin, duration_ms)

        self.mqtt.publish("squirrel/fire", '{"state": "fired"}')
