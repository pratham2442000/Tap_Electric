import os
from typing import Generator
from app.config import settings
from app.core.logging import logger

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, declarative_base
    
    Base = declarative_base()
    
    db_url = settings.DATABASE_URL
    is_sqlite = db_url.startswith("sqlite")
    
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    
    try:
        engine = create_engine(
            db_url,
            connect_args=connect_args,
            pool_pre_ping=not is_sqlite,
        )
        # Test connection
        with engine.connect() as conn:
            pass
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        if is_sqlite:
            Base.metadata.create_all(bind=engine)
    except Exception as conn_err:
        logger.info(f"Primary DATABASE_URL not accessible ({conn_err}). Falling back to local SQLite database.")
        test_url = settings.DATABASE_TEST_URL
        engine = create_engine(test_url, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)

except ImportError:
    engine = None
    SessionLocal = None
    Base = object
    logger.warning("SQLAlchemy not installed. Install requirements to enable database features.")


def init_db(target_engine=None):
    """Initializes or resets database tables on target or active engine."""
    global engine, SessionLocal
    if target_engine is not None:
        engine = target_engine
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    if engine is not None and Base is not object:
        Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


