import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.ml.metrics import evaluate_predictions
from app.validation.format_validator import EVSEFormatValidator
from app.core.logging import logger

SAMPLE_PREDS = [
    "DE*ISE*E1234567910",
    "+49*123*12345678910",
    "NL*TNM*E00101",
    "DE*1SE*E1234567910",  # 1 character error (1 instead of I)
    "FR*ALL*E55011"
]

SAMPLE_REFS = [
    "DE*ISE*E1234567910",
    "+49*123*12345678910",
    "NL*TNM*E00101",
    "DE*ISE*E1234567910",
    "FR*ALL*E55011"
]

def main():
    logger.info("Benchmarking EVSE OCR Performance...")
    metrics = evaluate_predictions(SAMPLE_PREDS, SAMPLE_REFS)
    
    validator = EVSEFormatValidator()
    valid_count = 0
    for pred in SAMPLE_PREDS:
        res = validator.validate_and_normalize(pred)
        if res["is_valid"]:
            valid_count += 1
            
    logger.info(f"Exact Match Accuracy : {metrics['exact_match_accuracy'] * 100:.2f}%")
    logger.info(f"Character Error Rate : {metrics['cer'] * 100:.2f}%")
    logger.info(f"Word Error Rate      : {metrics['wer'] * 100:.2f}%")
    logger.info(f"Format Validity Rate : {(valid_count / len(SAMPLE_PREDS)) * 100:.2f}%")

if __name__ == "__main__":
    main()
