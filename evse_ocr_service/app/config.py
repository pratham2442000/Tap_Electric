import os
from typing import Optional, Set
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "EVSE OCR Recovery Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Database (PostgreSQL + PostGIS)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/evse_ocr_db"
    )
    DATABASE_TEST_URL: str = os.getenv(
        "DATABASE_TEST_URL", "sqlite:///./test.db"
    )
    
    # Object Storage (S3 / MinIO / Local)
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # 's3' or 'local'
    S3_ENDPOINT_URL: Optional[str] = os.getenv("S3_ENDPOINT_URL", None)
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "evse-scan-telemetry")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    AWS_REGION: str = os.getenv("AWS_REGION", "eu-central-1")
    LOCAL_STORAGE_DIR: str = os.getenv("LOCAL_STORAGE_DIR", "./data/raw")
    
    # ML / Model Parameters
    MODEL_CHECKPOINT: str = os.getenv("MODEL_CHECKPOINT", "microsoft/trocr-base-printed")
    MODEL_MAX_LENGTH: int = 40
    MODEL_NUM_BEAMS: int = 5
    MODEL_NO_REPEAT_NGRAM_SIZE: int = 3
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "auto")  # 'cuda', 'cpu', 'mps', 'auto'
    
    # Active Learning & Validation Thresholds
    CONFIDENCE_THRESHOLD: float = 0.85
    SPATIAL_VALIDATION_RADIUS_KM: float = 0.5  # 500 meters search radius
    ALLOWED_IMAGE_TYPES: Set[str] = {"image/jpeg", "image/png", "image/heic", "image/webp"}
    
    model_config = {
        "case_sensitive": True,
        "env_file": ".env",
    }

settings = Settings()
