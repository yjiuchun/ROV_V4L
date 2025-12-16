import os
import sys
scripts_dir = '/home/yjc/Project/rov_ws/src/vision4location/scripts'
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from extrema_detector import ExtremaDetector

class ExtrameLightnessDetector:
    def __init__(self):
        self.extrema_detector = ExtremaDetector()

    def detect(self, img):
        extrema = self.extrema_detector.detect(img)
        max_positions = extrema["max_positions"]
        return max_positions