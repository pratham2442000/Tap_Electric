import unittest
from app.ml.metrics import levenshtein_distance, calculate_cer, calculate_wer, evaluate_predictions

class TestMetrics(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("DE*ISE*E123", "DE*ISE*E123"), 0)
        self.assertEqual(levenshtein_distance("DE*1SE*E123", "DE*ISE*E123"), 1)

    def test_calculate_cer(self):
        preds = ["DE*1SE*E123"]
        refs = ["DE*ISE*E123"]
        cer = calculate_cer(preds, refs)
        self.assertAlmostEqual(cer, 1.0 / 11.0, places=4)

    def test_evaluate_predictions(self):
        preds = ["DE*ISE*E123", "NL*TNM*E001"]
        refs = ["DE*ISE*E123", "NL*TNM*E002"]
        metrics = evaluate_predictions(preds, refs)
        self.assertEqual(metrics["exact_match_accuracy"], 0.5)
        self.assertTrue(metrics["cer"] > 0.0)

if __name__ == "__main__":
    unittest.main()
