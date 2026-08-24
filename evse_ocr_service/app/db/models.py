from datetime import datetime
import uuid

try:
    from sqlalchemy import Column, String, Float, DateTime, Boolean, Integer, ForeignKey, JSON
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
    from sqlalchemy.orm import relationship
    from app.core.database import Base
    
    UUIDType = PG_UUID(as_uuid=True)
    JSONType = JSONB
except ImportError:
    Base = object
    Column = String = Float = DateTime = Boolean = Integer = ForeignKey = JSON = None
    relationship = None
    UUIDType = None
    JSONType = None

if Base is not object:
    class ScanEvent(Base):
        """Represents a failed or degraded scan telemetry capture event."""
        __tablename__ = 'scan_events'
        
        id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
        timestamp_utc = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
        
        # Spatial Anchors
        latitude = Column(Float, nullable=False, index=True)
        longitude = Column(Float, nullable=False, index=True)
        location_accuracy_m = Column(Float, nullable=True)
        
        # Environmental Context (ambient lux, camera ISO, user device ID, partial decode)
        environmental_context = Column(JSON, nullable=True)
        
        # Infrastructure Pointer to Object Storage
        s3_object_uri = Column(String(512), nullable=False, unique=True)
        
        # Active Learning & Pipeline State Machine
        is_annotated = Column(Boolean, default=False, index=True, nullable=False)
        training_iteration = Column(Integer, default=0, nullable=False)
        is_valid_format = Column(Boolean, nullable=True)
        
        # Relationships
        annotations = relationship("TextAnnotation", back_populates="scan_event", cascade="all, delete-orphan")

    class TextAnnotation(Base):
        """Stores extracted EVSE ID hypotheses from models or human auditors."""
        __tablename__ = 'text_annotations'
        
        id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
        scan_event_id = Column(UUIDType, ForeignKey('scan_events.id', ondelete='CASCADE'), nullable=False)
        
        # Extracted alphanumeric string (e.g., DE*ISE*E1234567910)
        extracted_text = Column(String(255), nullable=True)
        
        # Provenance tracking: 'human_auditor', 'trocr_v1.0', 'easyocr_baseline'
        provenance = Column(String(100), nullable=False, index=True)
        confidence_score = Column(Float, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
        
        scan_event = relationship("ScanEvent", back_populates="annotations")

    class ChargingStation(Base):
        """Master record of physical charging station infrastructure."""
        __tablename__ = 'charging_stations'
        
        id = Column(String(64), primary_key=True)  # e.g., 'AMS-STATION-001'
        operator_id = Column(String(10), nullable=False, index=True)  # e.g., 'TNM', 'ISE', 'ALL'
        name = Column(String(255), nullable=False)
        latitude = Column(Float, nullable=False, index=True)
        longitude = Column(Float, nullable=False, index=True)
        address = Column(String(255), nullable=True)
        country_code = Column(String(2), nullable=False, index=True)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
        
        evse_assets = relationship("EVSEAsset", back_populates="station", cascade="all, delete-orphan")

    class EVSEAsset(Base):
        """Individual physical charging outlet / EVSE ID."""
        __tablename__ = 'evse_assets'
        
        id = Column(String(64), primary_key=True)  # e.g., 'DE*ISE*E1234567910'
        station_id = Column(String(64), ForeignKey('charging_stations.id', ondelete='CASCADE'), nullable=False)
        standard_type = Column(String(20), nullable=False)  # 'ISO_15118' or 'DIN_91286'
        is_active = Column(Boolean, default=True, nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
        
        station = relationship("ChargingStation", back_populates="evse_assets")
