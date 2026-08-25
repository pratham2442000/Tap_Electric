import unittest
from PIL import Image, ImageDraw
from app.ml.augmentation import DegradationSimulator


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.simulator = DegradationSimulator()

    def test_simulator_output_shape(self):
        """Verify simulator preserves original image size and outputs RGB mode."""
        img = Image.new("RGB", (200, 80), color=(255, 255, 255))
        augmented = self.simulator.apply(img)
        self.assertEqual(augmented.size, (200, 80))
        self.assertEqual(augmented.mode, "RGB")

    def test_simulator_various_dimensions(self):
        """Verify simulator handles small, standard, and large aspect ratios."""
        test_sizes = [(100, 40), (420, 160), (800, 300)]
        for size in test_sizes:
            img = Image.new("RGB", size, color=(240, 240, 240))
            augmented = self.simulator.apply(img)
            self.assertEqual(augmented.size, size)
            self.assertEqual(augmented.mode, "RGB")

    def test_simulator_rgba_input_conversion(self):
        """Verify simulator cleanly converts RGBA input images to RGB."""
        img_rgba = Image.new("RGBA", (300, 100), color=(255, 255, 255, 255))
        augmented = self.simulator.apply(img_rgba)
        self.assertEqual(augmented.mode, "RGB")
        self.assertEqual(augmented.size, (300, 100))

    def test_simulator_pil_fallback_execution(self):
        """Verify simulator PIL fallback executes reliably when pipeline is explicitly None."""
        sim_fallback = DegradationSimulator()
        sim_fallback.pipeline = None
        
        img = Image.new("RGB", (250, 100), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)
        draw.text((20, 40), "NL*TNM*E123456", fill=(0, 0, 0))
        
        degraded = sim_fallback.apply(img)
        self.assertIsInstance(degraded, Image.Image)
        self.assertEqual(degraded.size, (250, 100))
        self.assertEqual(degraded.mode, "RGB")


if __name__ == "__main__":
    unittest.main()

