from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    version: str
    database_connected: bool
    model_loaded: bool
