import os
from typing import Generator
from app.config import settings
from app.core.logging import logger

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, declarative_base
    
    db_url = settings.DATABASE_URL
    is_sqlite = db_url.startswith("sqlite")
    
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    
    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=not is_sqlite,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except ImportError:
    engine = None
    SessionLocal = None
    Base = object
    logger.warning("SQLAlchemy not installed. Install requirements to enable database features.")

def get_db() -> Generator:
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
