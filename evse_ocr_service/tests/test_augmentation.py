import unittest
from PIL import Image
from app.ml.augmentation import DegradationSimulator

class TestAugmentation(unittest.TestCase):
    def test_simulator_output_shape(self):
        img = Image.new("RGB", (200, 80), color=(255, 255, 255))
        simulator = DegradationSimulator()
        augmented = simulator.apply(img)
        self.assertEqual(augmented.size, (200, 80))
        self.assertEqual(augmented.mode, "RGB")

if __name__ == "__main__":
    unittest.main()
