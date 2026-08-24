import uuid
import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from pydantic import ValidationError
from app.config import settings
from app.schemas.telemetry import ScanTelemetry, ScanTelemetryResponse
from app.workers.background_tasks import persist_scan_data
from app.core.logging import logger

router = APIRouter(prefix="/telemetry", tags=["Telemetry Ingestion"])

@router.post(
    "/failed_scans",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ScanTelemetryResponse,
    summary="Ingest failed QR scan attempts from mobile client"
)
async def ingest_failed_scan(
    background_tasks: BackgroundTasks,
    image_payload: UploadFile = File(..., description="Raw optical scan image buffer"),
    telemetry_data: str = Form(..., description="JSON string containing environmental & location context")
):
    """
    Ingests failed QR scan attempts from the mobile client.
    Validates MIME type, deserializes JSON metadata, and asynchronously offloads
    object storage and relational DB writes to background workers.
    """
    if image_payload.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported Media Type '{image_payload.content_type}'. Allowed types: {list(settings.ALLOWED_IMAGE_TYPES)}"
        )
        
    try:
        parsed_metadata = json.loads(telemetry_data)
        validated_telemetry = ScanTelemetry(**parsed_metadata)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed telemetry payload: {str(e)}"
        )
        
    trace_id = str(uuid.uuid4())
    file_buffer = await image_payload.read()
    
    # Offload I/O operations to background task
    background_tasks.add_task(
        persist_scan_data,
        trace_id,
        file_buffer,
        validated_telemetry,
        image_payload.content_type
    )
    
    return {
        "status": "accepted",
        "trace_id": trace_id,
        "message": "Telemetry successfully queued for background persistence."
    }
