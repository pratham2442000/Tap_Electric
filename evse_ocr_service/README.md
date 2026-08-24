# End-to-End Machine Learning Service for Degraded EVSE ID Recognition and Recovery

A production-grade, distributed machine learning service engineered to capture failed/degraded QR code scan attempts from mobile charging applications, persist telemetry and imagery, and continuously train a Transformer-based Optical Character Recognition (**TrOCR**) model to recover human-readable EVSE (Electric Vehicle Supply Equipment) identifiers.

---

## System Architecture Overview

```
                          +------------------------------------------+
                          |   Mobile Client (Tap Electric App)       |
                          +------------------------------------------+
                                  |                         |
               1. Failed QR Scan  |                         | 2. Direct Fallback
               + Telemetry        v                         v    Inference Request
               +-----------------------------+   +-----------------------------+
               | /api/v1/telemetry/          |   | /api/v1/inference/          |
               | failed_scans                |   | recover                     |
               +-----------------------------+   +-----------------------------+
                              |                                 |
                     (202 Accepted)                             v
                              |                   +----------------------------+
                              v                   | TrOCR Inference Engine     |
               +-----------------------------+    | (ViT Encoder + LM Decoder) |
               | Background Worker           |    +----------------------------+
               +-----------------------------+                  |
                     |                 |                        v
                     v                 v          +----------------------------+
         +---------------+    +----------------+  | EVSE Format Validator      |
         | AWS S3 /      |    | PostgreSQL +   |  | (DIN SPEC 91286 /          |
         | Object Storage|    | PostGIS DB     |  |  ISO 15118-1 + Heuristics) |
         | (Raw Images)  |    | (Spatial Meta) |  +----------------------------+
         +---------------+    +----------------+                |
                                       ^                        v
                                       |          +----------------------------+
                                       +----------| Geospatial Cross-Validator |
                                (Spatial Check)   | (Haversine & PostGIS)      |
                                                  +----------------------------+
```

---

## Key Features

1. **High-Concurrency Telemetry Ingestion API (FastAPI)**
   - Asynchronous multipart upload endpoint handling raw binary image frames alongside environmental telemetry (ambient lux, camera ISO, GPS coordinates, device UUID, and partial QR decodes).
   - Instant `202 Accepted` response with non-blocking background workers offloading persistence.

2. **Bifurcated Storage Architecture**
   - **Object Storage (S3 / MinIO / Local)**: Secure, infinitely scalable raw image buffer store.
   - **Relational Spatial Database (PostgreSQL + PostGIS)**: Stores normalized `ScanEvent`, `TextAnnotation`, `ChargingStation`, and `EVSEAsset` models with spatial indices.

3. **Microsoft TrOCR Vision-Language Engine**
   - Vision Transformer (ViT) patch-based visual encoder coupled with an autoregressive language model decoder.
   - Configured with Beam Search ($N=5$) and strict decoding constraints to eliminate hallucinations.

4. **Synthetic Degradation Simulation & Continuous Training (MLOps)**
   - `DegradationSimulator` using `albumentations` reproducing real-world failure modes: focus/motion blur, extreme exposure clipping, UV fading / thermal ink washout, physical scratches, and sensor noise.
   - Hugging Face `Seq2SeqTrainer` fine-tuning pipeline computing Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein distance matrices.

5. **Deterministic Regulatory Validation & Geospatial Cross-Referencing**
   - Regex engines strictly verifying **DIN SPEC 91286** (EVCO-ID) and **ISO 15118-1** (EMA-ID) formats.
   - Heuristic optical confusion matrix correction (e.g. `O` $	o$ `0`, `I` $	o$ `1`, `+` prefix formatting).
   - PostGIS distance checking verifying that predicted identifiers correspond to registered assets near the user's GPS coordinates.

---

## Directory Structure

```
.
├── Dockerfile                       # Multi-stage container definition
├── docker-compose.yml               # Local stack (FastAPI, PostGIS, MinIO)
├── requirements.txt                 # Pinned dependencies
├── pyproject.toml                   # Project metadata and pytest configuration
├── alembic.ini                      # Database migration configuration
├── alembic/
│   ├── env.py                       # Alembic migration environment
│   └── versions/
│       └── 0001_initial_schema.py   # Initial database schema DDL
├── app/
│   ├── main.py                      # FastAPI application instance & lifecycle
│   ├── config.py                    # Environment settings via Pydantic
│   ├── api/
│   │   ├── router.py                # Top-level API router
│   │   └── v1/
│   │       ├── telemetry.py         # POST /api/v1/telemetry/failed_scans
│   │       ├── inference.py         # POST /api/v1/inference/recover
│   │       └── annotations.py       # Human audit & ground truth endpoints
│   ├── core/
│   │   ├── database.py              # SQLAlchemy engine & session dependency
│   │   ├── storage.py               # S3 and Local storage abstraction
│   │   └── logging.py               # Structured logger
│   ├── db/
│   │   ├── models.py                # ScanEvent, TextAnnotation, ChargingStation, EVSEAsset
│   │   └── repositories/
│   │       ├── scan_repository.py   # CRUD for scan events & annotations
│   │       └── station_repository.py# PostGIS geospatial queries
│   ├── schemas/
│   │   ├── telemetry.py             # Telemetry request/response models
│   │   ├── inference.py             # Inference request/response models
│   │   ├── annotation.py            # Annotation schemas
│   │   └── common.py                # Health & pagination schemas
│   ├── ml/
│   │   ├── inference.py             # TrOCR Inference Engine wrapper
│   │   ├── augmentation.py          # DegradationSimulator (Albumentations)
│   │   ├── synthetic_generator.py   # Synthetic EVSE ID sticker generator
│   │   ├── training.py              # Seq2SeqTrainer fine-tuning pipeline
│   │   ├── active_learning.py       # Active learning triage logic
│   │   └── metrics.py               # Levenshtein distance, CER & WER
│   ├── validation/
│   │   ├── format_validator.py      # DIN SPEC 91286 / ISO 15118-1 regex validator
│   │   └── geospatial.py            # Haversine distance & candidate matching
│   └── workers/
│       └── background_tasks.py      # Background persistence worker
├── scripts/
│   ├── seed_database.py             # Seeds sample European charging networks
│   ├── generate_synthetic_data.py   # Batch generates synthetic degraded stickers
│   ├── run_training.py              # CLI to trigger TrOCR fine-tuning
│   └── evaluate_model.py            # Benchmark script (CER, WER, regex pass rate)
└── tests/
    ├── conftest.py
    ├── test_format_validator.py     # Regulatory regex & heuristic tests
    ├── test_geospatial.py           # Haversine & radius filtering tests
    ├── test_metrics.py              # CER & WER mathematical validation
    ├── test_augmentation.py         # Augmentation dimension & mode tests
    ├── test_api_telemetry.py        # Telemetry schema tests
    └── test_inference_engine.py     # Inference engine tests
```

---

## Quickstart & Installation

### Option 1: Using Docker Compose (Recommended)

To start the full stack including the FastAPI API, PostgreSQL with PostGIS, and MinIO S3 object storage:

```bash
# Clone the repository and navigate into the directory
cd evse_ocr_service

# Copy the sample environment file
cp .env.example .env

# Launch the services
docker-compose up --build -d
```

Once running:
- **API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **MinIO S3 Console**: `http://localhost:9001` (User: `minioadmin` / Pass: `minioadmin`)
- **PostGIS Database**: `localhost:5432` (`evse_ocr_db`)

### Option 2: Local Python Virtual Environment

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run database migrations / seed sample data
python3 scripts/seed_database.py

# 4. Start the FastAPI development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Running Tests

Execute the automated test suite with `pytest` or Python's standard `unittest`:

```bash
python3 -m unittest discover tests
```

---

## API Reference

### 1. Ingest Failed Scan Telemetry
`POST /api/v1/telemetry/failed_scans`

**Multipart Form Data:**
- `image_payload`: Binary image file (JPEG, PNG, HEIC)
- `telemetry_data`: JSON string:
```json
{
  "timestamp_utc": "2026-08-24T10:00:00Z",
  "latitude": 52.379189,
  "longitude": 4.899431,
  "location_accuracy_m": 4.5,
  "ambient_lux": 85.0,
  "camera_iso": 400,
  "user_device_id": "550e8400-e29b-41d4-a716-446655440000",
  "partial_decode": "DE*ISE*"
}
```

**Response (`202 Accepted`):**
```json
{
  "status": "accepted",
  "trace_id": "8b58a5c3-3c97-4b72-a16f-998811223344",
  "message": "Telemetry successfully queued for background persistence."
}
```

---

### 2. Direct EVSE ID Recovery & Inference
`POST /api/v1/inference/recover`

**Multipart Form Data:**
- `image_payload`: Binary image file
- `latitude`: `52.379189` (Optional)
- `longitude`: `4.899431` (Optional)
- `apply_heuristics`: `true`

**Response (`200 OK`):**
```json
{
  "extracted_text": "DE*ISE*E1234567910",
  "normalized_id": "DE*ISE*E1234567910",
  "is_valid": true,
  "detected_standard": "ISO_15118",
  "confidence_score": 0.9624,
  "parsed_components": {
    "country_code": "DE",
    "operator_id": "ISE",
    "outlet_id": "E1234567910",
    "standard_type": "ISO_15118"
  },
  "geospatial_match": true,
  "candidate_stations": [
    {
      "station_id": "BER-CS-002",
      "station_name": "Ionity - Berlin Hauptbahnhof",
      "operator_id": "ISE",
      "evse_id": "DE*ISE*E1234567910",
      "standard_type": "ISO_15118",
      "distance_meters": 12.4
    }
  ],
  "trace_id": "c71a3962-e64e-4f05-89f5-efc4e334df5a"
}
```

---

## MLOps Training & Active Learning Workflow

1. **Generate Synthetic Pre-training Data:**
   ```bash
   python3 scripts/generate_synthetic_data.py --count 500 --output_dir ./data/synthetic
   ```

2. **Trigger Model Fine-Tuning:**
   ```bash
   python3 scripts/run_training.py --epochs 10 --batch_size 16 --lr 2e-5
   ```

3. **Evaluate Model Accuracy (CER / WER / Regulatory Pass Rate):**
   ```bash
   python3 scripts/evaluate_model.py
   ```
