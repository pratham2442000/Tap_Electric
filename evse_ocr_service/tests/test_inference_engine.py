import unittest
from PIL import Image
from app.ml.inference import EVSEInferenceEngine

class TestInferenceEngine(unittest.TestCase):
    def test_engine_extraction(self):
        engine = EVSEInferenceEngine()
        img = Image.new("RGB", (300, 100), color=(240, 240, 240))
        text, conf = engine.extract_text(img)
        self.assertIsInstance(text, str)
        self.assertIsInstance(conf, float)
        self.assertTrue(0.0 <= conf <= 1.0)

if __name__ == "__main__":
    unittest.main()
