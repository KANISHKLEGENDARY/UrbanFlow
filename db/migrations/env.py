"""
UrbanFlow -- Alembic Environment Configuration

Supports both async (online) and offline migration modes.
The async engine is configured from config.DATABASE_URL.
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so we can import config/models.
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as app_config  # noqa: E402
from db.models import Base  # noqa: E402

# ---------------------------------------------------------------------------
# Alembic Config object — gives access to values in alembic.ini
# ---------------------------------------------------------------------------
alembic_cfg = context.config

# Interpret the config file for Python logging.
if alembic_cfg.config_file_name is not None:
    fileConfig(alembic_cfg.config_file_name)

# Override the sqlalchemy.url from our application config
alembic_cfg.set_main_option("sqlalchemy.url", app_config.DATABASE_URL)

# Metadata used by autogenerate
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline mode — generates SQL script without connecting to the database
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates the SQL statements to stdout without needing a live database.
    """
    url = alembic_cfg.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode — connects to the database asynchronously and runs migrations
# ---------------------------------------------------------------------------
def do_run_migrations(connection: Connection) -> None:
    """Configure context with a live connection and run migrations."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations inside a connection."""
    connectable = async_engine_from_config(
        alembic_cfg.get_section(alembic_cfg.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
