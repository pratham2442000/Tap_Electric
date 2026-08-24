import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.annotation import TextAnnotationCreate, TextAnnotationResponse
from app.db.repositories.scan_repository import ScanRepository
from app.core.logging import logger

router = APIRouter(prefix="/annotations", tags=["Active Learning & Human Auditing"])

@router.get(
    "/pending",
    summary="Retrieve unannotated scan events requiring human ground truth"
)
async def get_pending_audits(
    limit: int = 50,
    db: Optional[Session] = Depends(get_db)
):
    if not db:
        return {"items": [], "total": 0}
    repo = ScanRepository(db)
    events = repo.get_unannotated_events(limit=limit)
    return {
        "items": [
            {
                "id": str(e.id),
                "timestamp_utc": e.timestamp_utc,
                "latitude": e.latitude,
                "longitude": e.longitude,
                "s3_object_uri": e.s3_object_uri,
                "environmental_context": e.environmental_context,
            }
            for e in events
        ],
        "total": len(events)
    }

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=TextAnnotationResponse,
    summary="Submit ground-truth annotation for a scan event"
)
async def submit_annotation(
    payload: TextAnnotationCreate,
    db: Optional[Session] = Depends(get_db)
):
    if not db:
        raise HTTPException(status_code=500, detail="Database not configured")
        
    repo = ScanRepository(db)
    annotation = repo.add_annotation(
        scan_event_id=payload.scan_event_id,
        extracted_text=payload.extracted_text,
        provenance=payload.provenance,
        confidence_score=payload.confidence_score
    )
    if not annotation:
        raise HTTPException(status_code=400, detail="Failed to save annotation")
    return annotation
