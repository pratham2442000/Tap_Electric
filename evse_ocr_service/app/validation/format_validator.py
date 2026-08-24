import re
from typing import Dict, Any, Optional

class EVSEFormatValidator:
    """
    Validates and normalizes extracted EVSE ID strings against international standards:
    - DIN SPEC 91286 / EVCO-ID (e.g. +49*123*12345678910 or +49-123-12345678910)
    - ISO 15118-1 / EMA-ID (e.g. DE*ISE*E1234567910 or DEISEE1234567910)
    """
    def __init__(self):
        # ISO 15118-1 / EMA-ID Validation Pattern:
        # 2-letter Country Code, optional delimiter (*), 3-char Operator ID, same delimiter, 'E', and outlet ID
        self.iso_regex = re.compile(
            r"^[A-Z]{2}(\*?)[A-Z0-9]{3}(?:\1)[E][A-Z0-9][A-Z0-9\*]{0,30}$",
            re.IGNORECASE
        )
        
        # DIN SPEC 91286 / EVCO-ID Validation Pattern:
        # Optional '+' prefix, 2-letter/digit country code, delimiter (* or -), 3-char operator ID, delimiter, outlet ID (1-30 chars)
        self.din_regex = re.compile(
            r"^(?:\+?[A-Z0-9]{2}[\*\-])[A-Z0-9]{3}[\*\-][A-Z0-9\*\-]{1,30}$",
            re.IGNORECASE
        )

    def clean_and_normalize(self, ocr_text: str, apply_heuristics: bool = True) -> str:
        """
        Cleans raw OCR output, strips whitespace, converts to uppercase,
        and applies domain-specific heuristic corrections.
        """
        if not ocr_text:
            return ""
            
        # Strip all whitespace commonly hallucinated by OCR models
        clean_text = re.sub(r"\s+", "", ocr_text).upper()
        
        # Standardize delimiter variations (e.g., bullet points or dots to asterisks)
        clean_text = clean_text.replace("•", "*").replace("·", "*")
        
        if apply_heuristics:
            # Heuristic 1: If string starts with '+', convert 'O' to '0' in country code prefix (e.g., +4O -> +40)
            if clean_text.startswith("+"):
                prefix = clean_text[:3].replace("O", "0")
                clean_text = prefix + clean_text[3:]
                
            # Heuristic 2: ISO 15118 identifiers start with a 2-letter alpha country code.
            if len(clean_text) >= 2 and not clean_text.startswith("+"):
                first_two = clean_text[:2]
                if first_two.startswith("0") and first_two[1].isalpha():
                    clean_text = "D" + clean_text[1:]
                    
        return clean_text

    def parse_components(self, clean_text: str) -> Dict[str, Optional[str]]:
        """
        Decomposes a valid EVSE ID into Country Code, Operator ID, and Outlet ID.
        """
        components = {
            "country_code": None,
            "operator_id": None,
            "outlet_id": None,
            "standard_type": "UNKNOWN"
        }
        
        if "*" in clean_text or "-" in clean_text:
            delimiter = "*" if "*" in clean_text else "-"
            parts = clean_text.split(delimiter)
            if len(parts) >= 3:
                components["country_code"] = parts[0].lstrip("+")
                components["operator_id"] = parts[1]
                components["outlet_id"] = delimiter.join(parts[2:])
            elif len(parts) == 2:
                components["operator_id"] = parts[0]
                components["outlet_id"] = parts[1]
        else:
            if len(clean_text) >= 6 and clean_text[:2].isalpha():
                components["country_code"] = clean_text[:2]
                components["operator_id"] = clean_text[2:5]
                components["outlet_id"] = clean_text[5:]
                
        return components

    def validate_and_normalize(self, ocr_text: str, apply_heuristics: bool = True) -> Dict[str, Any]:
        """
        Main validation entrypoint.
        """
        clean_text = self.clean_and_normalize(ocr_text, apply_heuristics=apply_heuristics)
        
        is_iso = bool(self.iso_regex.match(clean_text))
        is_din = bool(self.din_regex.match(clean_text))
        
        detected_standard = "ISO_15118" if is_iso else ("DIN_91286" if is_din else "UNKNOWN")
        components = self.parse_components(clean_text)
        components["standard_type"] = detected_standard
        
        return {
            "is_valid": is_iso or is_din,
            "detected_standard": detected_standard,
            "normalized_id": clean_text,
            "parsed_components": components,
            "raw_text": ocr_text
        }
