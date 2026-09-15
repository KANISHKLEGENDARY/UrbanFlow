"""
UrbanFlow -- Async PostgreSQL Database Setup

The API uses this module to initialize SQLAlchemy's async engine and create
tables when PostgreSQL is available. Connection failures are handled by the
database service so local development can continue without a running database.
"""

import config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db.models import Base


class DatabaseManager:
    """Small wrapper around SQLAlchemy async engine/sessionmaker."""

    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.available = False

    def configure(self) -> bool:
        if not config.DATABASE_ENABLED:
            self.available = False
            return False

        if self.engine is None:
            self.engine = create_async_engine(
                config.DATABASE_URL,
                echo=False,
                pool_pre_ping=True,
            )
            self.session_factory = async_sessionmaker(
                self.engine,
                expire_on_commit=False,
                class_=AsyncSession,
            )
        return True

    async def create_schema(self) -> None:
        if not self.configure() or self.engine is None:
            return

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.available = True

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()
        self.available = False

    def session(self) -> AsyncSession:
        if self.session_factory is None:
            raise RuntimeError("Database is not configured.")
        return self.session_factory()


database_manager = DatabaseManager()
