import unittest
import numpy as np
from include.Correlations import Correlator

class TestCorrelator(unittest.TestCase):

    def test_Numpy(self):
        Corr = Correlator()
        Corr.loadTestData()
        Corr.setLengthsAndSteps(5,1)
        Corr.setMode("Numpy")
        Corr.Correlate()
        data, hist = Corr.getHistograms()
        out = np.array([0, 2, 3, 0, 1, 2, 2, 0, 1])
        self.assertTrue(np.array_equal(data, out))

    def test_Numpy_jit(self):
        Corr = Correlator()
        Corr.loadTestData()
        Corr.setLengthsAndSteps(5, 1)
        Corr.setMode("Numpy")
        Corr.setJit(True)
        Corr.Correlate()
        data, hist = Corr.getHistograms()
        out = np.array([0, 2, 3, 0, 1, 2, 2, 0, 1])
        self.assertTrue(np.array_equal(data, out))

    """
    TBD
    def test_Numpy_Parallel(self):
        Corr = Correlator()
        Corr.loadTestData()
        Corr.setLengthsAndSteps(5, 1)
        Corr.setMode("NumpyParallel")
        Corr.Correlate()
        data, hist = Corr.getHistograms()
        out = np.array([0, 2, 3, 0, 1, 2, 2, 0, 1])
        self.assertTrue(np.array_equal(data, out))
    """

    def test_Loop(self):
        Corr = Correlator()
        Corr.loadTestData()
        Corr.setLengthsAndSteps(5, 1)
        Corr.setMode("Loop")
        Corr.Correlate()
        data, hist = Corr.getHistograms()
        out = np.array([0, 2, 3, 0, 1, 2, 2, 0, 1])
        self.assertTrue(np.array_equal(data, out))

    def test_Loop_Jit(self):
        Corr = Correlator()
        Corr.loadTestData()
        Corr.setLengthsAndSteps(5, 1)
        Corr.setMode("Loop")
        Corr.setJit(True)
        Corr.Correlate()
        data, hist = Corr.getHistograms()
        out = np.array([0, 2, 3, 0, 1, 2, 2, 0, 1])
        self.assertTrue(np.array_equal(data, out))

    def test_Cuda(self):
        Corr = Correlator()
        Corr.loadTestData()
        Corr.setLengthsAndSteps(5, 1)
        Corr.setMode("CUDA")
        Corr.Correlate()
        data, hist = Corr.getHistograms()
        out = np.array([0, 2, 3, 0, 1, 2, 2, 0, 1])
        self.assertTrue(np.array_equal(data, out))

class TestInterface(unittest.TestCase):

    def test_test(self):
        pass

if __name__ == "__main__":
    unittest.main()
