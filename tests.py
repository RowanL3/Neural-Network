import unittest

from neural_network import loss

class TestNeuralNetwork(unittest.TestCase):
    def test_loss(self):
        self.assertEqual(loss([1,1],[1,1]), 0)
        self.assertEqual(loss([2,1],[1,1]), 1)
        self.assertEqual(loss([5,1],[1,1]), 16)


if __name__ == '__main__':
    unittest.main()
