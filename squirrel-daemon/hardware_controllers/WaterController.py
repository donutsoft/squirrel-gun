from .MqttClient import MqttClient
import time

class WaterController:
    def __init__(self):
        self.mqtt = MqttClient()

    def startWatering(self, duration):
        print(f"Starting watering for {duration} seconds.")
        self.mqtt.publish("zigbee2mqtt/hose/set", '{"state": "ON"}')
        time.sleep(duration)
        self.mqtt.publish("zigbee2mqtt/hose/set", '{"state": "OFF"}')
        # Code to activate water pump for 'duration' seconds

    def stopWatering(self):
        print("Stopping watering.")
        self.mqtt.publish("zigbee2mqtt/hose/set", '{"state": "OFF"}')
