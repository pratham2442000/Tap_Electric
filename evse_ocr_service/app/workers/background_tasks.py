import uuid
from datetime import datetime
from typing import Optional
from app.schemas.telemetry import ScanTelemetry
from app.core.storage import get_storage_service
from app.core.database import SessionLocal
from app.db.repositories.scan_repository import ScanRepository
from app.validation.format_validator import EVSEFormatValidator
from app.core.logging import logger

async def persist_scan_data(
    trace_id: str,
    image_bytes: bytes,
    meta: ScanTelemetry,
    content_type: str
):
    """
    Asynchronous background task to upload raw image to object storage,
    create relational metadata entry in PostgreSQL, and validate partial decodes.
    """
    try:
        logger.info(f"Executing background persistence for trace: {trace_id}")
        
        # 1. Upload to Object Storage
        storage = get_storage_service()
        s3_uri = await storage.upload_image(image_bytes, trace_id, content_type)
        
        # 2. Check initial format of partial decode if available
        validator = EVSEFormatValidator()
        is_valid = None
        if meta.partial_decode:
            val_res = validator.validate_and_normalize(meta.partial_decode)
            is_valid = val_res["is_valid"]
            
        # 3. Persist to PostgreSQL database
        if SessionLocal:
            db = SessionLocal()
            try:
                repo = ScanRepository(db)
                repo.create_scan_event(
                    event_id=uuid.UUID(trace_id),
                    timestamp_utc=meta.timestamp_utc,
                    latitude=meta.latitude,
                    longitude=meta.longitude,
                    location_accuracy_m=meta.location_accuracy_m,
                    environmental_context={
                        "ambient_lux": meta.ambient_lux,
                        "camera_iso": meta.camera_iso,
                        "user_device_id": meta.user_device_id,
                        "partial_decode": meta.partial_decode,
                    },
                    s3_object_uri=s3_uri,
                    is_valid_format=is_valid
                )
                logger.info(f"ScanEvent {trace_id} successfully persisted in PostgreSQL.")
            finally:
                db.close()
    except Exception as e:
        logger.error(f"Error in background persistence task for {trace_id}: {e}", exc_info=True)
