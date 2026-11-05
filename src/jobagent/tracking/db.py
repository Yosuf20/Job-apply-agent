"""Database engine and session setup for the tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from sqlalchemy import Engine, event
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.orm import Session, sessionmaker

from jobagent.config import Settings, get_settings
from jobagent.tracking.tables import Base

SessionFactory = sessionmaker[Session]
_SQLITE_FOREIGN_KEYS: Final[str] = "PRAGMA foreign_keys=ON"


def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(_SQLITE_FOREIGN_KEYS)
    finally:
        cursor.close()


def create_engine(settings: Settings | None = None) -> Engine:
    """Create the tracker engine from configured settings."""
    app_settings = settings or get_settings()
    database_path = Path(app_settings.database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqlalchemy_create_engine(f"sqlite:///{database_path}")
    _install_sqlite_foreign_keys(engine)
    return engine


def _install_sqlite_foreign_keys(engine: Engine) -> None:
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)


def create_session_factory(engine: Engine) -> SessionFactory:
    """Create a session factory bound to an engine."""
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all tracker tables and enable SQLite foreign keys."""
    _install_sqlite_foreign_keys(engine)
    with engine.connect() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql(_SQLITE_FOREIGN_KEYS)
    Base.metadata.create_all(engine)
