from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ScanTelemetry(BaseModel):
    timestamp_utc: datetime = Field(..., description="Timestamp of scan failure in UTC")
    latitude: float = Field(..., description="WGS 84 Latitude")
    longitude: float = Field(..., description="WGS 84 Longitude")
    location_accuracy_m: Optional[float] = Field(None, description="GPS uncertainty radius in meters")
    ambient_lux: Optional[float] = Field(None, description="Ambient light sensor level")
    camera_iso: Optional[int] = Field(None, description="Camera ISO setting")
    user_device_id: str = Field(..., description="Anonymized device UUID")
    partial_decode: Optional[str] = Field(None, description="Partial string captured before QR failure")

class ScanTelemetryResponse(BaseModel):
    status: str
    trace_id: str
    message: str
    s3_uri: Optional[str] = None
