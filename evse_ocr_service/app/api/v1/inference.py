import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.core.database import get_db
from app.schemas.inference import InferenceResponse, GeospatialCandidate
from app.ml.inference import get_inference_engine
from app.validation.format_validator import EVSEFormatValidator
from app.validation.geospatial import validate_candidate_locations
from app.db.repositories.station_repository import StationRepository
from app.core.logging import logger

router = APIRouter(prefix="/inference", tags=["Model Inference & Recovery"])

@router.post(
    "/recover",
    response_model=InferenceResponse,
    summary="Directly extract and validate EVSE ID from degraded image"
)
async def recover_degraded_evse_id(
    image_payload: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    apply_heuristics: bool = Form(True),
    db: Optional[Session] = Depends(get_db)
):
    """
    Performs real-time TrOCR inference on a degraded image of an EVSE identifier,
    validates the output against DIN SPEC 91286 / ISO 15118-1 standards,
    and cross-checks the result against registered charging station locations in PostGIS.
    """
    if image_payload.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported media format."
        )
        
    trace_id = str(uuid.uuid4())
    image_bytes = await image_payload.read()
    
    # 1. TrOCR Vision-Language Extraction
    engine = get_inference_engine()
    raw_extracted_text, confidence_score = engine.extract_text(image_bytes)
    
    # 2. Format Validation & Normalization
    validator = EVSEFormatValidator()
    validation_info = validator.validate_and_normalize(
        raw_extracted_text, apply_heuristics=apply_heuristics
    )
    
    # 3. Geospatial Cross-Validation
    geospatial_match = False
    candidate_list = []
    
    if latitude is not None and longitude is not None and db is not None:
        station_repo = StationRepository(db)
        candidates = station_repo.find_nearest_stations(
            latitude, longitude, max_radius_km=settings.SPATIAL_VALIDATION_RADIUS_KM
        )
        geo_validation = validate_candidate_locations(
            latitude, longitude, validation_info["normalized_id"], candidates,
            max_radius_km=settings.SPATIAL_VALIDATION_RADIUS_KM
        )
        geospatial_match = geo_validation["geospatial_match"]
        candidate_list = [
            GeospatialCandidate(**c) for c in candidates[:5]
        ]
        
    return InferenceResponse(
        extracted_text=raw_extracted_text,
        normalized_id=validation_info["normalized_id"],
        is_valid=validation_info["is_valid"],
        detected_standard=validation_info["detected_standard"],
        confidence_score=confidence_score,
        parsed_components=validation_info["parsed_components"],
        geospatial_match=geospatial_match,
        candidate_stations=candidate_list,
        trace_id=trace_id
    )
