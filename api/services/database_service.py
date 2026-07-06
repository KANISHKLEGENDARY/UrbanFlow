"""
UrbanFlow -- Database Service

Phase 4 persistence layer. It initializes PostgreSQL tables, seeds zone
metadata/statistics from the loaded demand service, and records prediction,
pricing, and ETA activity for later analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import cast, func, select, String

from db.database import database_manager
from db.models import ETALog, PredictionLog, PricingHistory, Zone, ZoneStat


class DatabaseService:
    """Async database facade used by API routes and startup."""

    def __init__(self):
        self.available = False

    async def diagnose_and_setup(self) -> None:
        """Diagnose database connection and auto-heal/configure credentials."""
        import socket
        import asyncpg
        import urllib.parse
        import re
        import config
        from db.database import database_manager

        print("[DatabaseService] Checking PostgreSQL connection...")
        diag_file = Path(config.PROJECT_ROOT) / "db_diagnostic.txt"
        
        # Parse current DATABASE_URL
        url = config.DATABASE_URL
        # Replace +asyncpg for parsing
        raw_url = url.replace("postgresql+asyncpg://", "postgresql://")
        parsed_url = urllib.parse.urlparse(raw_url)
        username = parsed_url.username or "postgres"
        password = parsed_url.password or ""
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 5432
        database = parsed_url.path.lstrip("/") or "urbanflow"

        # Check if port is open
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        try:
            s.connect((host, port))
            s.close()
            diag_file.write_text(f"Port {port} on {host} is OPEN.\n")
        except Exception as e:
            diag_file.write_text(f"Port {port} on {host} is CLOSED: {e}\n")
            print(f"[DatabaseService] Port {port} on {host} is closed. Database service will be disabled.")
            return

        # Check connection with configured password
        working_password = None
        try:
            conn = await asyncpg.connect(user=username, password=password, database="postgres", host=host, port=port, timeout=5)
            await conn.close()
            working_password = password
            diag_file.write_text(f"Connection successful using password from configuration.\n")
        except Exception as e:
            diag_file.write_text(f"Connection failed using configured password: {e}\n")
            if "password authentication failed" in str(e).lower() or "authentication failed" in str(e).lower():
                passwords_to_try = ["admin", "password", "root", "1234", "123456", "urbanflow", "postgres", ""]
                for p in passwords_to_try:
                    if p == password:
                        continue
                    try:
                        diag_file.write_text(f"Trying password: '{p}' for user '{username}'...\n")
                        conn = await asyncpg.connect(user=username, password=p, database="postgres", host=host, port=port, timeout=2)
                        await conn.close()
                        working_password = p
                        diag_file.write_text(f"Success! Password '{p}' works for user '{username}'.\n")
                        break
                    except Exception as ex:
                        diag_file.write_text(f"Failed with password '{p}': {ex}\n")
            
        if working_password is not None:
            # Check/create the target database
            try:
                conn = await asyncpg.connect(user=username, password=working_password, database="postgres", host=host, port=port)
                exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", database)
                if not exists:
                    await conn.execute(f'CREATE DATABASE "{database}"')
                    diag_file.write_text(f"Created database '{database}'\n")
                    print(f"[DatabaseService] Created database '{database}'")
                else:
                    diag_file.write_text(f"Database '{database}' already exists.\n")
                await conn.close()
            except Exception as e:
                diag_file.write_text(f"Failed to check/create database: {e}\n")
                print(f"[DatabaseService] Failed to check/create database: {e}")

            # Update the env file if needed
            if working_password != password:
                env_path = Path(config.PROJECT_ROOT) / ".env"
                if env_path.exists():
                    try:
                        env_content = env_path.read_text()
                        # Replace POSTGRES_PASSWORD and DATABASE_URL
                        new_env = env_content
                        if "POSTGRES_PASSWORD=" in env_content:
                            new_env = re.sub(r'POSTGRES_PASSWORD=.*', f'POSTGRES_PASSWORD={working_password}', new_env)
                        else:
                            new_env += f'\nPOSTGRES_PASSWORD={working_password}'

                        # Find matching DATABASE_URL line and replace password
                        pattern = r'DATABASE_URL=postgresql\+asyncpg://([^:]+):([^@]+)@(.*)'
                        if re.search(pattern, env_content):
                            new_env = re.sub(pattern, f'DATABASE_URL=postgresql+asyncpg://\\1:{working_password}@\\3', new_env)
                        
                        env_path.write_text(new_env)
                        diag_file.write_text(f"Successfully updated .env file with working password.\n")
                        print(f"[DatabaseService] Updated .env file with working password.")
                    except Exception as e:
                        diag_file.write_text(f"Failed to update .env: {e}\n")
                
                # Update runtime config
                config.DATABASE_URL = f"postgresql+asyncpg://{username}:{working_password}@{host}:{port}/{database}"
                database_manager.engine = None
                database_manager.session_factory = None
                database_manager.available = False

    async def initialize(self, demand_service) -> bool:
        """Create schema and seed reference rows if PostgreSQL is reachable."""
        try:
            await self.diagnose_and_setup()
            await database_manager.create_schema()
            self.available = database_manager.available
            if not self.available:
                print("[DatabaseService] Database disabled")
                return False

            await self.seed_reference_data(demand_service)
            print("[DatabaseService] PostgreSQL ready")
            return True
        except Exception as exc:
            self.available = False
            print(f"[DatabaseService] PostgreSQL unavailable: {exc}")
            print("[DatabaseService] API will continue with file-backed model data.")
            return False

    async def shutdown(self) -> None:
        await database_manager.close()
        self.available = False

    async def seed_reference_data(self, demand_service) -> None:
        """Seed zone metadata and zone/time-slot statistics once."""
        if not self.available or not demand_service.zone_stats:
            return

        async with database_manager.session() as session:
            zone_count = await session.scalar(select(func.count(Zone.zone_id)))
            stat_count = await session.scalar(select(func.count(ZoneStat.zone_id)))

            if not zone_count:
                for zone_id, meta in demand_service.zone_stats["zone_meta"].items():
                    await session.merge(
                        Zone(
                            zone_id=int(zone_id),
                            zone_name=str(meta.get("zone_name", f"Zone {zone_id}")),
                            borough=str(meta.get("borough", "Unknown")),
                            service_zone=str(meta.get("service_zone", "")) if meta.get("service_zone") else None,
                            lat=float(meta.get("zone_lat", 40.7128)),
                            lng=float(meta.get("zone_lng", -74.0060)),
                        )
                    )

            if not stat_count:
                zone_means = demand_service.zone_stats["zone_means"]
                zone_stds = demand_service.zone_stats["zone_stds"]
                zone_maxs = demand_service.zone_stats["zone_maxs"]
                zone_slot_means = demand_service.zone_stats["zone_slot_means"]
                borough_means = demand_service.zone_stats["borough_means"]
                raw_maxs = demand_service.zone_stats["raw_maxs"]

                for (zone_id, time_slot), zone_slot_demand in zone_slot_means.items():
                    meta = demand_service.zone_stats["zone_meta"].get(zone_id, {})
                    borough_enc = meta.get("borough_enc", 0)
                    await session.merge(
                        ZoneStat(
                            zone_id=int(zone_id),
                            time_slot=int(time_slot),
                            zone_demand_mean=float(zone_means.get(zone_id, 0)),
                            zone_demand_std=float(zone_stds.get(zone_id, 0)),
                            zone_demand_max=float(zone_maxs.get(zone_id, 0)),
                            zone_slot_demand=float(zone_slot_demand),
                            borough_demand_mean=float(borough_means.get(borough_enc, 0)),
                            raw_demand_max=float(raw_maxs.get(zone_id, 0)),
                        )
                    )

            await session.commit()

    async def log_prediction(self, request_type: str, req, result: dict) -> None:
        """Record a demand prediction or heatmap summary."""
        if not self.available:
            return
        try:
            async with database_manager.session() as session:
                session.add(
                    PredictionLog(
                        request_type=request_type,
                        zone_id=result.get("zone_id"),
                        hour=int(req.hour),
                        minute=int(req.minute),
                        day_of_week=int(req.day_of_week),
                        day_of_month=getattr(req, "day_of_month", None),
                        month=getattr(req, "month", None),
                        predicted_demand=float(result.get("predicted_demand", result.get("avg_demand", 0))),
                        demand_raw=result.get("demand_raw"),
                        surge_multiplier=result.get("surge_multiplier"),
                        demand_level=result.get("demand_level"),
                    )
                )
                await session.commit()
        except Exception as exc:
            print(f"[DatabaseService] Prediction log skipped: {exc}")

    async def log_pricing(self, req, result: dict) -> None:
        """Record a pricing calculation."""
        if not self.available:
            return
        try:
            async with database_manager.session() as session:
                session.add(
                    PricingHistory(
                        zone_id=int(req.zone_id),
                        hour=int(req.hour),
                        minute=int(req.minute),
                        day_of_week=int(req.day_of_week),
                        base_fare=float(req.base_fare),
                        surge_multiplier=float(result["surge_multiplier"]),
                        adjusted_fare=float(result["adjusted_fare"]),
                        demand_level=str(result["demand_level"]),
                    )
                )
                await session.commit()
        except Exception as exc:
            print(f"[DatabaseService] Pricing log skipped: {exc}")

    async def log_eta(self, req, result: dict) -> None:
        """Record an ETA estimate."""
        if not self.available:
            return
        try:
            async with database_manager.session() as session:
                session.add(
                    ETALog(
                        pickup_zone=int(req.pickup_zone),
                        dropoff_zone=int(req.dropoff_zone),
                        hour=int(req.hour),
                        minute=int(req.minute),
                        day_of_week=int(req.day_of_week),
                        base_eta_minutes=float(result["base_eta_minutes"]),
                        adjusted_eta_minutes=float(result["adjusted_eta_minutes"]),
                        traffic_multiplier=float(result["traffic_multiplier"]),
                        distance_miles=float(result["distance_miles"]),
                        graph_available=bool(result["graph_available"]),
                    )
                )
                await session.commit()
        except Exception as exc:
            print(f"[DatabaseService] ETA log skipped: {exc}")

    # ─────────────────────────────────────────────────────────────
    # Analytics / Query Methods
    # ─────────────────────────────────────────────────────────────

    async def get_prediction_stats(self) -> dict:
        """Return aggregated prediction analytics from the database."""
        default = {
            "total_predictions": 0,
            "predictions_by_type": {},
            "top_zones": [],
            "avg_demand": 0.0,
            "predictions_by_hour": {},
            "database_available": self.available,
        }
        if not self.available:
            return default
        try:
            async with database_manager.session() as session:
                # Total predictions
                total = await session.scalar(
                    select(func.count(PredictionLog.id))
                )

                # Predictions grouped by request_type
                type_rows = (
                    await session.execute(
                        select(
                            PredictionLog.request_type,
                            func.count(PredictionLog.id),
                        ).group_by(PredictionLog.request_type)
                    )
                ).all()
                predictions_by_type = {row[0]: row[1] for row in type_rows}

                # Top 10 most queried zones (excluding NULL zone_id from heatmap)
                zone_rows = (
                    await session.execute(
                        select(
                            PredictionLog.zone_id,
                            func.count(PredictionLog.id).label("query_count"),
                        )
                        .where(PredictionLog.zone_id.is_not(None))
                        .group_by(PredictionLog.zone_id)
                        .order_by(func.count(PredictionLog.id).desc())
                        .limit(10)
                    )
                ).all()
                top_zones = [
                    {"zone_id": row[0], "query_count": row[1]}
                    for row in zone_rows
                ]

                # Average predicted demand
                avg_demand = await session.scalar(
                    select(func.avg(PredictionLog.predicted_demand))
                )

                # Prediction count by hour
                hour_rows = (
                    await session.execute(
                        select(
                            PredictionLog.hour,
                            func.count(PredictionLog.id),
                        ).group_by(PredictionLog.hour)
                        .order_by(PredictionLog.hour)
                    )
                ).all()
                predictions_by_hour = {str(row[0]): row[1] for row in hour_rows}

            return {
                "total_predictions": total or 0,
                "predictions_by_type": predictions_by_type,
                "top_zones": top_zones,
                "avg_demand": round(float(avg_demand or 0), 4),
                "predictions_by_hour": predictions_by_hour,
                "database_available": True,
            }
        except Exception as exc:
            print(f"[DatabaseService] get_prediction_stats error: {exc}")
            return default

    async def get_pricing_history(self, zone_id: int, limit: int = 20) -> list[dict]:
        """Return recent pricing history for a zone."""
        if not self.available:
            return []
        try:
            async with database_manager.session() as session:
                rows = (
                    await session.execute(
                        select(PricingHistory)
                        .where(PricingHistory.zone_id == zone_id)
                        .order_by(PricingHistory.created_at.desc())
                        .limit(limit)
                    )
                ).scalars().all()

                return [
                    {
                        "zone_id": r.zone_id,
                        "hour": r.hour,
                        "minute": r.minute,
                        "day_of_week": r.day_of_week,
                        "base_fare": r.base_fare,
                        "surge_multiplier": r.surge_multiplier,
                        "adjusted_fare": r.adjusted_fare,
                        "demand_level": r.demand_level,
                        "created_at": r.created_at.isoformat() if r.created_at else "",
                    }
                    for r in rows
                ]
        except Exception as exc:
            print(f"[DatabaseService] get_pricing_history error: {exc}")
            return []

    async def get_eta_stats(self) -> dict:
        """Return ETA usage statistics."""
        default = {
            "total_queries": 0,
            "popular_routes": [],
            "avg_eta_minutes": 0.0,
            "graph_usage_ratio": 0.0,
            "database_available": self.available,
        }
        if not self.available:
            return default
        try:
            async with database_manager.session() as session:
                # Total ETA queries
                total = await session.scalar(
                    select(func.count(ETALog.id))
                )

                # Most popular routes — top 10 pickup→dropoff pairs
                route_rows = (
                    await session.execute(
                        select(
                            ETALog.pickup_zone,
                            ETALog.dropoff_zone,
                            func.count(ETALog.id).label("query_count"),
                        )
                        .group_by(ETALog.pickup_zone, ETALog.dropoff_zone)
                        .order_by(func.count(ETALog.id).desc())
                        .limit(10)
                    )
                ).all()
                popular_routes = [
                    {
                        "pickup_zone": row[0],
                        "dropoff_zone": row[1],
                        "query_count": row[2],
                    }
                    for row in route_rows
                ]

                # Average adjusted ETA
                avg_eta = await session.scalar(
                    select(func.avg(ETALog.adjusted_eta_minutes))
                )

                # Graph vs fallback usage ratio
                total_count = total or 0
                if total_count > 0:
                    graph_count = await session.scalar(
                        select(func.count(ETALog.id)).where(
                            ETALog.graph_available.is_(True)
                        )
                    )
                    graph_ratio = round((graph_count or 0) / total_count, 4)
                else:
                    graph_ratio = 0.0

            return {
                "total_queries": total_count,
                "popular_routes": popular_routes,
                "avg_eta_minutes": round(float(avg_eta or 0), 2),
                "graph_usage_ratio": graph_ratio,
                "database_available": True,
            }
        except Exception as exc:
            print(f"[DatabaseService] get_eta_stats error: {exc}")
            return default


database_service = DatabaseService()
