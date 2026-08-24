import uuid
from typing import List, Optional
from datetime import datetime
from app.db.models import ScanEvent, TextAnnotation

class ScanRepository:
    def __init__(self, db_session):
        self.db = db_session

    def create_scan_event(
        self,
        event_id: uuid.UUID,
        timestamp_utc: datetime,
        latitude: float,
        longitude: float,
        s3_object_uri: str,
        location_accuracy_m: Optional[float] = None,
        environmental_context: Optional[dict] = None,
        is_valid_format: Optional[bool] = None,
    ) -> Optional[ScanEvent]:
        if not self.db:
            return None
        event = ScanEvent(
            id=event_id,
            timestamp_utc=timestamp_utc,
            latitude=latitude,
            longitude=longitude,
            location_accuracy_m=location_accuracy_m,
            environmental_context=environmental_context,
            s3_object_uri=s3_object_uri,
            is_annotated=False,
            is_valid_format=is_valid_format
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def add_annotation(
        self,
        scan_event_id: uuid.UUID,
        extracted_text: str,
        provenance: str,
        confidence_score: Optional[float] = None
    ) -> Optional[TextAnnotation]:
        if not self.db:
            return None
        annotation = TextAnnotation(
            id=uuid.uuid4(),
            scan_event_id=scan_event_id,
            extracted_text=extracted_text,
            provenance=provenance,
            confidence_score=confidence_score
        )
        self.db.add(annotation)
        
        if provenance == "human_auditor":
            event = self.db.query(ScanEvent).filter(ScanEvent.id == scan_event_id).first()
            if event:
                event.is_annotated = True
                
        self.db.commit()
        self.db.refresh(annotation)
        return annotation

    def get_unannotated_events(self, limit: int = 100) -> List[ScanEvent]:
        if not self.db:
            return []
        return self.db.query(ScanEvent).filter(ScanEvent.is_annotated == False).limit(limit).all()
