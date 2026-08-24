import math
from typing import List, Dict, Any, Optional

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates in meters using the Haversine formula.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    
    return R * c

def validate_candidate_locations(
    client_lat: float,
    client_lon: float,
    predicted_evse_id: str,
    candidate_stations: List[Dict[str, Any]],
    max_radius_km: float = 0.5
) -> Dict[str, Any]:
    """
    Cross-references a predicted EVSE ID against physical charging assets near the client's GPS.
    """
    max_radius_m = max_radius_km * 1000.0
    exact_match = None
    operator_match = None
    
    for candidate in candidate_stations:
        dist = candidate.get("distance_meters", 0.0)
        if dist <= max_radius_m:
            if candidate.get("evse_id") == predicted_evse_id:
                exact_match = candidate
                break
            
            # Check if operator ID matches candidate
            predicted_upper = predicted_evse_id.upper()
            if candidate.get("operator_id") and candidate["operator_id"].upper() in predicted_upper:
                if not operator_match or dist < operator_match["distance_meters"]:
                    operator_match = candidate

    return {
        "geospatial_match": exact_match is not None,
        "exact_match": exact_match,
        "nearest_operator_match": operator_match,
        "client_coordinates": {"latitude": client_lat, "longitude": client_lon}
    }
