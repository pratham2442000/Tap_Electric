import os
import re
import tempfile
import unittest
from PIL import Image
from app.ml.synthetic_generator import SyntheticEVSEGenerator, COUNTRY_CODES, SAMPLE_OPERATORS
from app.ml.augmentation import DegradationSimulator


class TestSyntheticEVSEGenerator(unittest.TestCase):
    def setUp(self):
        self.simulator = DegradationSimulator()
        self.generator = SyntheticEVSEGenerator(simulator=self.simulator)

    def test_generate_random_evse_id_iso_15118(self):
        """Test ISO 15118 format ID generation: [Country][*]?[Operator][*]?E[Digits]"""
        for _ in range(50):
            evse_id = self.generator.generate_random_evse_id(standard="ISO_15118")
            self.assertIsInstance(evse_id, str)
            self.assertGreater(len(evse_id), 8)
            # Pattern: 2-letter country, optional delimiter, 3-char operator, optional delimiter, 'E', 6-10 digits
            match = re.match(r"^([A-Z]{2})\*?([A-Z0-9]{3})\*?E(\d{6,10})$", evse_id)
            self.assertIsNotNone(match, f"EVSE ID '{evse_id}' does not match ISO 15118 pattern")
            country, op, outlet = match.groups()
            self.assertIn(country, COUNTRY_CODES)
            self.assertIn(op, SAMPLE_OPERATORS)

    def test_generate_random_evse_id_din_91286(self):
        """Test DIN 91286 format ID generation: [Prefix][*|-]?[Operator][*|-]?[Digits]"""
        for _ in range(50):
            evse_id = self.generator.generate_random_evse_id(standard="DIN_91286")
            self.assertIsInstance(evse_id, str)
            self.assertGreater(len(evse_id), 6)
            match = re.match(r"^(\+49|[A-Z]{2})[\*\-]?([A-Z0-9]{3})[\*\-]?(\d{6,10})$", evse_id)
            self.assertIsNotNone(match, f"EVSE ID '{evse_id}' does not match DIN 91286 pattern")

    def test_render_sticker_image_default_dimensions(self):
        """Test render_sticker_image produces correct dimensions and RGB mode."""
        evse_id = "NL*TNM*E123456789"
        img = self.generator.render_sticker_image(evse_id)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (420, 160))
        self.assertEqual(img.mode, "RGB")

    def test_render_sticker_image_custom_dimensions_and_qr_toggle(self):
        """Test render_sticker_image with custom size and explicit QR options."""
        evse_id = "DE*FST*E987654321"
        img_qr = self.generator.render_sticker_image(evse_id, width=500, height=200, include_qr=True)
        self.assertEqual(img_qr.size, (500, 200))
        self.assertEqual(img_qr.mode, "RGB")

        img_no_qr = self.generator.render_sticker_image(evse_id, width=350, height=120, include_qr=False)
        self.assertEqual(img_no_qr.size, (350, 120))
        self.assertEqual(img_no_qr.mode, "RGB")

    def test_generate_pair(self):
        """Test generate_pair returns an image and corresponding ID string."""
        degraded_img, text = self.generator.generate_pair(standard="ISO_15118")
        self.assertIsInstance(degraded_img, Image.Image)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertEqual(degraded_img.mode, "RGB")

    def test_generate_dataset(self):
        """Test generating a synthetic dataset into a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            count = 5
            manifest = self.generator.generate_dataset(output_dir=tmpdir, count=count)
            
            self.assertEqual(len(manifest), count)
            for item in manifest:
                self.assertIn("image_path", item)
                self.assertIn("text", item)
                self.assertIn("standard", item)
                self.assertTrue(os.path.exists(item["image_path"]))
                
                # Verify that saved image can be opened and is valid
                with Image.open(item["image_path"]) as loaded_img:
                    self.assertEqual(loaded_img.mode, "RGB")
                    self.assertGreater(loaded_img.size[0], 0)
                    self.assertGreater(loaded_img.size[1], 0)


if __name__ == "__main__":
    unittest.main()
