import numpy as np


class DoublePhotonEvent:
    x1 = 0.
    y1 = 0.
    x2 = 0.
    y2 = 0.
    deltaT1 = 0.
    deltaT2 = 0.

    def __init__(self, x1, y1, x2, y2, delta_t1, delta_t2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.deltaT1 = delta_t1
        self.deltaT2 = delta_t2

    def get_position_one(self):
        return np.array([self.x1, self.y1])

    def get_position_two(self):
        return np.array([self.x2, self.y2])

    def get_time_one(self):
        return self.deltaT1

    def get_time_two(self):
        return self.deltaT2
