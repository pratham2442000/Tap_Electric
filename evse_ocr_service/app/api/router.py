from fastapi import APIRouter
from app.api.v1.telemetry import router as telemetry_router
from app.api.v1.inference import router as inference_router
from app.api.v1.annotations import router as annotations_router

api_router = APIRouter()
api_router.include_router(telemetry_router)
api_router.include_router(inference_router)
api_router.include_router(annotations_router)
