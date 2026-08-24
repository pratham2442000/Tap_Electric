import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.database import SessionLocal, engine, Base
from app.db.models import ChargingStation, EVSEAsset
from app.core.logging import logger

SAMPLE_STATIONS = [
    {
        "id": "AMS-CS-001",
        "operator_id": "TNM",
        "name": "Shell Recharge - Amsterdam Centraal",
        "latitude": 52.379189,
        "longitude": 4.899431,
        "address": "Stationsplein 1, 1012 AB Amsterdam",
        "country_code": "NL",
        "assets": [
            {"id": "NL*TNM*E00101", "standard_type": "ISO_15118"},
            {"id": "NL*TNM*E00102", "standard_type": "ISO_15118"},
        ]
    },
    {
        "id": "BER-CS-002",
        "operator_id": "ISE",
        "name": "Ionity - Berlin Hauptbahnhof",
        "latitude": 52.525589,
        "longitude": 13.369548,
        "address": "Europaplatz 1, 10557 Berlin",
        "country_code": "DE",
        "assets": [
            {"id": "DE*ISE*E1234567910", "standard_type": "ISO_15118"},
            {"id": "+49*123*12345678910", "standard_type": "DIN_91286"},
        ]
    },
    {
        "id": "ROT-CS-003",
        "operator_id": "FST",
        "name": "Fastned - Rotterdam Blaak",
        "latitude": 51.920555,
        "longitude": 4.489444,
        "address": "Blaak 100, 3011 TA Rotterdam",
        "country_code": "NL",
        "assets": [
            {"id": "NL*FST*E99001", "standard_type": "ISO_15118"},
            {"id": "NL*FST*E99002", "standard_type": "ISO_15118"},
        ]
    },
    {
        "id": "PAR-CS-004",
        "operator_id": "ALL",
        "name": "Allego - Paris Chatelet",
        "latitude": 48.858889,
        "longitude": 2.347222,
        "address": "Place du Châtelet, 75001 Paris",
        "country_code": "FR",
        "assets": [
            {"id": "FR*ALL*E55011", "standard_type": "ISO_15118"},
            {"id": "FR*ALL*E55012", "standard_type": "ISO_15118"},
        ]
    }
]

def seed():
    if not SessionLocal:
        logger.error("Database connection not configured.")
        return

    db = SessionLocal()
    try:
        logger.info("Creating database tables if not exist...")
        Base.metadata.create_all(bind=engine)
        
        for st_data in SAMPLE_STATIONS:
            existing = db.query(ChargingStation).filter(ChargingStation.id == st_data["id"]).first()
            if not existing:
                station = ChargingStation(
                    id=st_data["id"],
                    operator_id=st_data["operator_id"],
                    name=st_data["name"],
                    latitude=st_data["latitude"],
                    longitude=st_data["longitude"],
                    address=st_data["address"],
                    country_code=st_data["country_code"]
                )
                db.add(station)
                db.flush()
                
                for asset_data in st_data["assets"]:
                    asset = EVSEAsset(
                        id=asset_data["id"],
                        station_id=station.id,
                        standard_type=asset_data["standard_type"]
                    )
                    db.add(asset)
                logger.info(f"Seeded station: {st_data['name']} with {len(st_data['assets'])} EVSE assets.")
                
        db.commit()
        logger.info("Database seeding completed successfully.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed()
