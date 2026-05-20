import os
import subprocess
from .MqttClient import MqttClient

class WaterController:
    def __init__(self):
        self.mqtt = MqttClient()

    def startWatering(self, duration):
        print(f"Starting watering for {duration} seconds.")
        controller_path = os.path.join(os.path.dirname(__file__), "ValveController.sh")
        cmd = [controller_path, str(float(duration))]
        subprocess.Popen(cmd)

        self.mqtt.publish("squirrel/fire", '{"state": "fired"}')
