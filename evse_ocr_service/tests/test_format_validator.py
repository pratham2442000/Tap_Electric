import unittest
from app.validation.format_validator import EVSEFormatValidator

class TestEVSEFormatValidator(unittest.TestCase):
    def setUp(self):
        self.validator = EVSEFormatValidator()

    def test_iso_15118_valid_examples(self):
        cases = [
            "DE*ISE*E1234567910",
            "NL*TNM*E00101",
            "FR*ALL*E55011",
            "DEISEE1234567910"
        ]
        for c in cases:
            res = self.validator.validate_and_normalize(c)
            self.assertTrue(res["is_valid"], f"Failed on valid ISO case: {c}")
            self.assertEqual(res["detected_standard"], "ISO_15118")

    def test_din_91286_valid_examples(self):
        cases = [
            "+49*123*12345678910",
            "+49-123-12345678910",
            "+31*TNM*998877",
            "49*123*12345"
        ]
        for c in cases:
            res = self.validator.validate_and_normalize(c)
            self.assertTrue(res["is_valid"], f"Failed on valid DIN case: {c}")
            self.assertEqual(res["detected_standard"], "DIN_91286")

    def test_whitespace_and_heuristic_corrections(self):
        # Case with whitespace and confused 'O' in +4O prefix
        raw = " +4O * 123 * 12345678910 "
        res = self.validator.validate_and_normalize(raw, apply_heuristics=True)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["normalized_id"], "+40*123*12345678910")

    def test_invalid_strings(self):
        invalid_cases = [
            "INVALID_STRING_XYZ",
            "12345",
            "!",
            ""
        ]
        for c in invalid_cases:
            res = self.validator.validate_and_normalize(c)
            self.assertFalse(res["is_valid"], f"Should be invalid: {c}")

if __name__ == "__main__":
    unittest.main()
