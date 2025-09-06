import pigpio

class PanTiltController:
    MIN_US = 500
    MAX_US = 2500

    def __init__(self):
        self.pigpio = pigpio.pi()
        self.pan_pin =  18
        self.tilt_pin = 19

    def _setAngle(self, pin, angle):
        angle = max(0, min(180, angle))  # clamp
        pulse = self.MIN_US + (angle / 180.0) * (self.MAX_US - self.MIN_US)
        self.pigpio.set_servo_pulsewidth(pin, pulse)

    
    def setPan(self, angle):
        # Invert pan so UI left/right match physical movement
        self._setAngle(self.pan_pin, 180 - angle)

    def setTilt(self, angle):
        self._setAngle(self.tilt_pin, angle)

    def setPanTilt(self, pan, tilt):
        self.setPan(pan)
        self.setTilt(tilt)
