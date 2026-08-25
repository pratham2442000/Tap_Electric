import io
import os
import json
import uuid
import tempfile
import unittest
from datetime import datetime
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.config import settings
from app.core.database import Base, get_db, init_db
from app.db.models import ChargingStation, EVSEAsset, ScanEvent, TextAnnotation
from app.ml.synthetic_generator import SyntheticEVSEGenerator
from app.ml.augmentation import DegradationSimulator
from app.ml.inference import get_inference_engine
from app.ml.training import EVSETrainingPipeline
from app.ml.metrics import evaluate_predictions
from app.workers.background_tasks import persist_scan_data
from app.schemas.telemetry import ScanTelemetry


class TestFullSystemE2E(unittest.TestCase):
    """
    Full End-to-End System Integration Test:
    Simulates a real-world flow where:
    1. A fake EV driver at a charging station gets a degraded/bad sticker image.
    2. The mobile client ingests telemetry to POST /api/v1/telemetry/failed_scans (202 Accepted).
    3. Background worker persists raw image to storage and metadata to DB.
    4. The client requests real-time OCR recovery at POST /api/v1/inference/recover.
    5. The system performs OCR extraction, format validation, and geospatial candidate matching.
    6. The active learning workflow fetches unannotated scans from GET /api/v1/annotations/pending
       and submits ground-truth via POST /api/v1/annotations.
    7. The local dataset is loaded into the ML training pipeline, preprocessed, and evaluated.
    """

    @classmethod
    def setUpClass(cls):
        # Create isolated temporary directory for local storage and SQLite DB
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.storage_dir = os.path.join(cls.temp_dir.name, "storage")
        cls.db_path = os.path.join(cls.temp_dir.name, "test_e2e.db")
        os.makedirs(cls.storage_dir, exist_ok=True)

        settings.LOCAL_STORAGE_DIR = cls.storage_dir
        settings.STORAGE_TYPE = "local"
        settings.DATABASE_URL = f"sqlite:///{cls.db_path}"

        # Setup SQLite Database Engine and sync with app database module
        cls.engine = create_engine(f"sqlite:///{cls.db_path}", connect_args={"check_same_thread": False})
        init_db(cls.engine)
        cls.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)


        # Seed physical charging station infrastructure into database
        cls.seed_station_id = "AMS-CS-001"
        cls.seed_operator = "TNM"
        cls.seed_evse_id = "NL*TNM*E102938475"
        cls.station_lat = 52.379189
        cls.station_lon = 4.899431

        db = cls.SessionLocal()
        try:
            station = ChargingStation(
                id=cls.seed_station_id,
                operator_id=cls.seed_operator,
                name="Amsterdam Central Fast Charger",
                latitude=cls.station_lat,
                longitude=cls.station_lon,
                country_code="NL"
            )
            asset = EVSEAsset(
                id=cls.seed_evse_id,
                station_id=cls.seed_station_id,
                standard_type="ISO_15118",
                is_active=True
            )
            db.add(station)
            db.add(asset)
            db.commit()
        finally:
            db.close()

        # Override get_db dependency in FastAPI app
        def override_get_db():
            db_session = cls.SessionLocal()
            try:
                yield db_session
            finally:
                db_session.close()

        app.dependency_overrides[get_db] = override_get_db
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()
        cls.temp_dir.cleanup()

    def test_full_system_user_journey(self):
        """Execute the full end-to-end simulated workflow."""
        # -------------------------------------------------------------
        # STEP 1: Generate degraded image from a fake user's camera
        # -------------------------------------------------------------
        generator = SyntheticEVSEGenerator()
        simulator = DegradationSimulator()

        # Render clean sticker for the registered station and apply degradation
        clean_img = generator.render_sticker_image(self.seed_evse_id, width=440, height=160, include_qr=True)
        degraded_img = simulator.apply(clean_img)

        img_byte_arr = io.BytesIO()
        degraded_img.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()

        self.assertGreater(len(image_bytes), 0, "Bad image buffer should be non-empty")

        # -------------------------------------------------------------
        # STEP 2: Fake user submits failed scan telemetry (POST /telemetry/failed_scans)
        # -------------------------------------------------------------
        user_lat = 52.379200  # ~1.5 meters from station
        user_lon = 4.899440
        device_id = str(uuid.uuid4())

        telemetry_payload = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "latitude": user_lat,
            "longitude": user_lon,
            "location_accuracy_m": 4.5,
            "ambient_lux": 60.0,
            "camera_iso": 640,
            "user_device_id": device_id,
            "partial_decode": "NL*TNM*"
        }

        response = self.client.post(
            "/api/v1/telemetry/failed_scans",
            files={"image_payload": ("camera_scan.png", image_bytes, "image/png")},
            data={"telemetry_data": json.dumps(telemetry_payload)}
        )

        self.assertEqual(response.status_code, 202, f"Expected 202 Accepted, got: {response.text}")
        resp_json = response.json()
        self.assertEqual(resp_json["status"], "accepted")
        self.assertIn("trace_id", resp_json)
        trace_id = resp_json["trace_id"]

        # -------------------------------------------------------------
        # STEP 3: Verify Background Persistence (Storage & DB)
        # Note: TestClient automatically executes background_tasks on response completion
        # -------------------------------------------------------------
        db = self.SessionLocal()
        try:
            event = db.query(ScanEvent).filter(ScanEvent.id == uuid.UUID(trace_id)).first()
            self.assertIsNotNone(event, "ScanEvent must be persisted in database by background worker")
            self.assertEqual(event.latitude, user_lat)
            self.assertEqual(event.environmental_context["user_device_id"], device_id)
            self.assertFalse(event.is_annotated, "New event should not be annotated yet")
            
            # Verify file exists on local storage disk (stripping file:// URI prefix)
            actual_file_path = event.s3_object_uri.replace("file://", "")
            self.assertTrue(os.path.exists(actual_file_path), f"Image file must exist on disk at {actual_file_path}")
        finally:
            db.close()


        # -------------------------------------------------------------
        # STEP 4: Real-time OCR Recovery (POST /inference/recover)
        # -------------------------------------------------------------
        inference_resp = self.client.post(
            "/api/v1/inference/recover",
            files={"image_payload": ("camera_scan.png", image_bytes, "image/png")},
            data={
                "latitude": str(user_lat),
                "longitude": str(user_lon),
                "apply_heuristics": "true"
            }
        )

        self.assertEqual(inference_resp.status_code, 200, f"Expected 200 OK, got: {inference_resp.text}")
        inf_data = inference_resp.json()
        self.assertIn("normalized_id", inf_data)
        self.assertIn("confidence_score", inf_data)
        self.assertIn("candidate_stations", inf_data)
        self.assertTrue(len(inf_data["candidate_stations"]) > 0, "Nearby charging station must be found")
        self.assertEqual(inf_data["candidate_stations"][0]["station_id"], self.seed_station_id)

        # -------------------------------------------------------------
        # STEP 5: Active Learning & Human Audit Queue
        # -------------------------------------------------------------
        pending_resp = self.client.get("/api/v1/annotations/pending")
        self.assertEqual(pending_resp.status_code, 200)
        pending_items = pending_resp.json()["items"]
        self.assertTrue(any(item["id"] == trace_id for item in pending_items), "Event must appear in pending queue")

        # Submit human ground-truth annotation
        annotation_payload = {
            "scan_event_id": trace_id,
            "extracted_text": self.seed_evse_id,
            "provenance": "human_auditor",
            "confidence_score": 1.0
        }
        submit_resp = self.client.post("/api/v1/annotations", json=annotation_payload)
        self.assertEqual(submit_resp.status_code, 201)

        # Verify event is now marked as annotated
        db = self.SessionLocal()
        try:
            updated_event = db.query(ScanEvent).filter(ScanEvent.id == uuid.UUID(trace_id)).first()
            self.assertTrue(updated_event.is_annotated, "Event should now be marked as annotated")
        finally:
            db.close()

        # -------------------------------------------------------------
        # STEP 6: Simulated Local Dataset Loading & Metrics Evaluation
        # -------------------------------------------------------------
        # Save a sample pair to the local simulated dataset
        dataset_sample_path = os.path.join(self.storage_dir, "synthetic_eval_001.png")
        degraded_img.save(dataset_sample_path)

        local_dataset_records = [
            {
                "image_path": dataset_sample_path,
                "text": self.seed_evse_id
            }
        ]

        engine = get_inference_engine()
        pipeline = EVSETrainingPipeline(model_engine=engine)
        
        # Verify prepare_dataset runs on the local data records
        processed_data = pipeline.prepare_dataset(local_dataset_records, simulator)
        self.assertEqual(len(processed_data), 1)
        self.assertIn("pixel_values", processed_data[0])
        self.assertIn("labels", processed_data[0])

        # Evaluate predictions metric calculation
        pred_texts = [self.seed_evse_id]
        ref_texts = [self.seed_evse_id]
        metrics = evaluate_predictions(pred_texts, ref_texts)
        self.assertEqual(metrics["exact_match_accuracy"], 1.0)
        self.assertEqual(metrics["cer"], 0.0)
        self.assertEqual(metrics["wer"], 0.0)


if __name__ == "__main__":
    unittest.main()
