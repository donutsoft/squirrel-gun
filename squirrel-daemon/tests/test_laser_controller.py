import unittest

from hardware_controllers.LaserController import LaserController


class FakeSerialController:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.commands = []
        self.events = []

    def command(self, name, *args):
        self.commands.append((name, *args))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response

    def trace_event(self, message):
        self.events.append(message)


class LaserControllerTests(unittest.TestCase):
    def make_laser(self, responses):
        laser = LaserController.__new__(LaserController)
        laser._pin = 3
        laser._serial = FakeSerialController(responses)
        return laser

    def test_turn_on_retries_once_after_missing_response(self):
        laser = self.make_laser([TimeoutError("no response"), "OK"])

        laser.turn_on()

        self.assertEqual(laser._serial.commands, [("ON", 3), ("ON", 3)])
        self.assertIn("laser-on-retry pin=3", laser._serial.events)

    def test_turn_on_does_not_retry_an_explicit_rejection(self):
        laser = self.make_laser([RuntimeError("ERROR")])

        with self.assertRaisesRegex(RuntimeError, "ERROR"):
            laser.turn_on()

        self.assertEqual(laser._serial.commands, [("ON", 3)])

    def test_turn_on_propagates_second_timeout(self):
        laser = self.make_laser([TimeoutError("first"), TimeoutError("second")])

        with self.assertRaisesRegex(TimeoutError, "second"):
            laser.turn_on()

        self.assertEqual(laser._serial.commands, [("ON", 3), ("ON", 3)])


if __name__ == "__main__":
    unittest.main()
