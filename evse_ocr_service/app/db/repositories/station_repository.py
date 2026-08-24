from typing import List, Optional
from app.db.models import ChargingStation, EVSEAsset
from app.validation.geospatial import haversine_distance

class StationRepository:
    def __init__(self, db_session):
        self.db = db_session

    def find_nearest_stations(
        self, latitude: float, longitude: float, max_radius_km: float = 1.0
    ) -> List[dict]:
        if not self.db:
            return []
        
        lat_delta = max_radius_km / 111.0
        lon_delta = max_radius_km / (111.0 * max(0.1, abs(latitude)))
        
        stations = self.db.query(ChargingStation).filter(
            ChargingStation.latitude >= latitude - lat_delta,
            ChargingStation.latitude <= latitude + lat_delta,
            ChargingStation.longitude >= longitude - lon_delta,
            ChargingStation.longitude <= longitude + lon_delta,
        ).all()
        
        results = []
        for st in stations:
            dist = haversine_distance(latitude, longitude, st.latitude, st.longitude)
            if dist <= max_radius_km * 1000.0:
                for asset in st.evse_assets:
                    results.append({
                        "station_id": st.id,
                        "station_name": st.name,
                        "operator_id": st.operator_id,
                        "evse_id": asset.id,
                        "standard_type": asset.standard_type,
                        "distance_meters": round(dist, 2),
                        "latitude": st.latitude,
                        "longitude": st.longitude
                    })
        
        results.sort(key=lambda x: x["distance_meters"])
        return results

    def get_asset_by_id(self, evse_id: str) -> Optional[EVSEAsset]:
        if not self.db:
            return None
        return self.db.query(EVSEAsset).filter(EVSEAsset.id == evse_id).first()
