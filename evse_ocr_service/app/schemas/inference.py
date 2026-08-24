from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class GeospatialCandidate(BaseModel):
    station_id: str
    station_name: str
    operator_id: str
    evse_id: str
    standard_type: str
    distance_meters: float

class InferenceRequestOptions(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    apply_geospatial_validation: bool = True
    apply_heuristic_corrections: bool = True

class InferenceResponse(BaseModel):
    extracted_text: str
    normalized_id: str
    is_valid: bool
    detected_standard: str
    confidence_score: float
    parsed_components: Dict[str, Any] = {}
    geospatial_match: bool = False
    candidate_stations: List[GeospatialCandidate] = []
    trace_id: str
