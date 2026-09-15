"""
Initialize the UrbanFlow PostgreSQL database.

Creates the `urbanflow` database if missing, then creates all ORM tables.
Reads connection settings from .env via config.py.

Usage:
    python scripts/init_database.py
"""

import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
import config
from db.database import database_manager


def _parse_database_url() -> dict:
    """Parse DATABASE_URL into asyncpg connection kwargs."""
    raw = config.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(raw)
    return {
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": (parsed.path or "/urbanflow").lstrip("/") or "urbanflow",
    }


async def ensure_database_exists() -> None:
    """Create the application database when connecting to PostgreSQL succeeds."""
    params = _parse_database_url()
    db_name = params.pop("database")

    try:
        conn = await asyncpg.connect(database="postgres", **params)
    except Exception as exc:
        print(f"[init_database] Cannot connect to PostgreSQL: {exc}")
        print(
            "[init_database] Fix DATABASE_URL in .env, or start Docker PostgreSQL:\n"
            "    docker compose up db -d"
        )
        sys.exit(1)

    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"[init_database] Created database '{db_name}'")
        else:
            print(f"[init_database] Database '{db_name}' already exists")
    finally:
        await conn.close()


async def create_tables() -> None:
    """Create SQLAlchemy ORM tables."""
    await database_manager.create_schema()
    if database_manager.available:
        print("[init_database] Tables ready")
    else:
        print("[init_database] Failed to create tables")
        sys.exit(1)


async def main() -> None:
    print(f"[init_database] Target: {config.DATABASE_URL.split('@')[-1]}")
    await ensure_database_exists()
    await create_tables()
    await database_manager.close()
    print("[init_database] Done")


if __name__ == "__main__":
    asyncio.run(main())
