import os
import sys
import subprocess
from .MqttClient import MqttClient
import time

class WaterController:
    def __init__(self):
        self.mqtt = MqttClient()

    def startWatering(self, duration):
        print(f"Starting watering for {duration} seconds.")
        # Launch a new Python process to run the valve controller with duration
        script_path = os.path.join(os.path.dirname(__file__), "ValveController.py")
        cmd = [sys.executable, script_path, str(float(duration))]
        try:
            # Start the process without blocking
            subprocess.Popen(cmd)
        except Exception as e:
            print(f"Failed to start ValveController: {e}")

        self.mqtt.publish("squirrel/fire", '{"state": "fired"}')