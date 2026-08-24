from typing import Optional
import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class TextAnnotationCreate(BaseModel):
    scan_event_id: uuid.UUID
    extracted_text: str
    provenance: str = Field("human_auditor", description="Source of annotation")
    confidence_score: Optional[float] = 1.0

class TextAnnotationResponse(BaseModel):
    id: uuid.UUID
    scan_event_id: uuid.UUID
    extracted_text: str
    provenance: str
    confidence_score: Optional[float]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

