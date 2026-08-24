from typing import List, Dict, Any, Optional
from app.config import settings
from app.validation.format_validator import EVSEFormatValidator
from app.core.logging import logger

class ActiveLearningManager:
    """
    Coordinates the active learning feedback loop:
    1. Evaluates incoming scan events against confidence and format compliance thresholds.
    2. Flags uncertain or invalid scans for human auditing.
    3. Triggers fine-tuning runs when sufficient human ground truth has accumulated.
    """
    def __init__(self, validator: Optional[EVSEFormatValidator] = None):
        self.validator = validator or EVSEFormatValidator()
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD

    def evaluate_scan_for_triage(
        self,
        extracted_text: str,
        confidence_score: float,
        geospatial_match: bool
    ) -> Dict[str, Any]:
        validation_result = self.validator.validate_and_normalize(extracted_text)
        is_valid = validation_result["is_valid"]
        
        # Determine if human audit is required
        needs_human_audit = (
            (not is_valid) or
            (confidence_score < self.confidence_threshold) or
            (not geospatial_match)
        )
        
        triage_reason = []
        if not is_valid:
            triage_reason.append("FORMAT_VIOLATION")
        if confidence_score < self.confidence_threshold:
            triage_reason.append("LOW_CONFIDENCE")
        if not geospatial_match:
            triage_reason.append("GEOSPATIAL_MISMATCH")

        return {
            "needs_human_audit": needs_human_audit,
            "triage_reason": triage_reason,
            "validation_result": validation_result,
            "confidence_score": confidence_score
        }
