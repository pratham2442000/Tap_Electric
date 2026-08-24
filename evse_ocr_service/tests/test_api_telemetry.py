import unittest
from datetime import datetime
from app.schemas.telemetry import ScanTelemetry

class TestTelemetrySchema(unittest.TestCase):
    def test_valid_telemetry_deserialization(self):
        data = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "latitude": 52.379189,
            "longitude": 4.899431,
            "location_accuracy_m": 5.0,
            "ambient_lux": 150.0,
            "camera_iso": 200,
            "user_device_id": "550e8400-e29b-41d4-a716-446655440000",
            "partial_decode": "DE*ISE*"
        }
        telemetry = ScanTelemetry(**data)
        self.assertEqual(telemetry.latitude, 52.379189)
        self.assertEqual(telemetry.partial_decode, "DE*ISE*")

if __name__ == "__main__":
    unittest.main()
