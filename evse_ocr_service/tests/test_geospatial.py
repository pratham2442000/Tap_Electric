import unittest
from app.validation.geospatial import haversine_distance, validate_candidate_locations

class TestGeospatialValidation(unittest.TestCase):
    def test_haversine_same_point(self):
        dist = haversine_distance(52.379189, 4.899431, 52.379189, 4.899431)
        self.assertAlmostEqual(dist, 0.0, places=2)

    def test_haversine_amsterdam_to_berlin(self):
        # Amsterdam to Berlin is ~575 km
        dist = haversine_distance(52.379189, 4.899431, 52.525589, 13.369548)
        self.assertTrue(550000 < dist < 600000)

    def test_validate_candidate_locations_match(self):
        candidates = [
            {
                "station_id": "AMS-001",
                "station_name": "Amsterdam Centraal",
                "operator_id": "TNM",
                "evse_id": "NL*TNM*E00101",
                "distance_meters": 45.0
            }
        ]
        res = validate_candidate_locations(
            client_lat=52.3792,
            client_lon=4.8995,
            predicted_evse_id="NL*TNM*E00101",
            candidate_stations=candidates,
            max_radius_km=0.5
        )
        self.assertTrue(res["geospatial_match"])
        self.assertIsNotNone(res["exact_match"])

    def test_validate_candidate_locations_too_far(self):
        candidates = [
            {
                "station_id": "BER-002",
                "station_name": "Berlin Hbf",
                "operator_id": "ISE",
                "evse_id": "DE*ISE*E1234567910",
                "distance_meters": 575000.0
            }
        ]
        res = validate_candidate_locations(
            client_lat=52.3792,
            client_lon=4.8995,
            predicted_evse_id="DE*ISE*E1234567910",
            candidate_stations=candidates,
            max_radius_km=0.5
        )
        self.assertFalse(res["geospatial_match"])

if __name__ == "__main__":
    unittest.main()
