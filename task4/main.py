import time
import argparse
import logging

import cv2


class Sensor:
    def get(self):
        raise NotImplementedError("Subprocess must implement method get()")


class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._delta = 0
    
    def get(self) -> int:
        time.sleep(self._delay)
        self._delta += 1
        return self._delta


class SensorCam(Sensor):
    def __init__(self, camera_descriptor: str, resolution: str):
        self._resolution = tuple(map(int, resolution.split("x")))

    def __del__(self):
        pass

    def get(self):
        return 0
    

class WindowImage:
    def __init__(self, freqency: float):
        pass

    def __del__(self):
        pass

    def show(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--device", type=str, help="Name of your video device", default="/dev/video0")
    parser.add_argument("-r", "--resolution", type=str, help="Video resolution", default="1920x1080")
    parser.add_argument("-f", "--freq", type=float, help="Video flow frequency", default=1)

    args = parser.parse_args()
    print(args)
