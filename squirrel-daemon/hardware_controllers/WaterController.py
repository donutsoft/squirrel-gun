import os
import sys
import subprocess
from .MqttClient import MqttClient

class WaterController:
    def __init__(self):
        self.mqtt = MqttClient()

    def startWatering(self, duration):
        print(f"Starting watering for {duration} seconds.")
        # Launch a new Python process to run the valve controller with duration
        daemon_root = os.path.dirname(os.path.dirname(__file__))
        cmd = [sys.executable, "-m", "hardware_controllers.ValveController", str(float(duration))]
        subprocess.Popen(cmd, cwd=daemon_root)

        self.mqtt.publish("squirrel/fire", '{"state": "fired"}')
