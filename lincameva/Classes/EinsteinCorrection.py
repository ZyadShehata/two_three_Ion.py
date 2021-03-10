from photonscore.python.ann.evaluate import evaluate
import numpy as np

class EinsteinCorrection:
    def __init__(self):
        pass

    @staticmethod
    def correct_single(x,y,a,detectorNetwork):
        values = np.array([x,y,a]).T.copy()
        return evaluate([3,9,9], np.fromfile("./clbr/PX29110.double", dtype=np.float64), values)
