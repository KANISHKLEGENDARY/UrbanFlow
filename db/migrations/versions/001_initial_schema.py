"""Initial schema — all five UrbanFlow tables.

Revision ID: 001
Revises: None
Create Date: 2026-07-01

Tables created:
    - zones            (reference zone metadata)
    - zone_stats       (zone × time-slot aggregated statistics)
    - predictions_log  (demand prediction audit trail)
    - pricing_history  (surge pricing audit trail)
    - eta_log          (ETA estimation audit trail)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── zones ─────────────────────────────────────────────────────
    op.create_table(
        "zones",
        sa.Column("zone_id", sa.Integer, primary_key=True),
        sa.Column("zone_name", sa.String(160), nullable=False),
        sa.Column("borough", sa.String(80), nullable=False),
        sa.Column("service_zone", sa.String(80), nullable=True),
        sa.Column("lat", sa.Float, nullable=False),
        sa.Column("lng", sa.Float, nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # ── zone_stats ────────────────────────────────────────────────
    op.create_table(
        "zone_stats",
        sa.Column("zone_id", sa.Integer, primary_key=True),
        sa.Column("time_slot", sa.Integer, primary_key=True),
        sa.Column("zone_demand_mean", sa.Float, nullable=False),
        sa.Column("zone_demand_std", sa.Float, nullable=False),
        sa.Column("zone_demand_max", sa.Float, nullable=False),
        sa.Column("zone_slot_demand", sa.Float, nullable=False),
        sa.Column("borough_demand_mean", sa.Float, nullable=False),
        sa.Column("raw_demand_max", sa.Float, nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # ── predictions_log ───────────────────────────────────────────
    op.create_table(
        "predictions_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("request_type", sa.String(40), nullable=False),
        sa.Column("zone_id", sa.Integer, nullable=True),
        sa.Column("hour", sa.Integer, nullable=False),
        sa.Column("minute", sa.Integer, nullable=False),
        sa.Column("day_of_week", sa.Integer, nullable=False),
        sa.Column("day_of_month", sa.Integer, nullable=True),
        sa.Column("month", sa.Integer, nullable=True),
        sa.Column("predicted_demand", sa.Float, nullable=False),
        sa.Column("demand_raw", sa.Float, nullable=True),
        sa.Column("surge_multiplier", sa.Float, nullable=True),
        sa.Column("demand_level", sa.String(30), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_predictions_log_created_at", "predictions_log", ["created_at"])

    # ── pricing_history ───────────────────────────────────────────
    op.create_table(
        "pricing_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("zone_id", sa.Integer, nullable=False),
        sa.Column("hour", sa.Integer, nullable=False),
        sa.Column("minute", sa.Integer, nullable=False),
        sa.Column("day_of_week", sa.Integer, nullable=False),
        sa.Column("base_fare", sa.Float, nullable=False),
        sa.Column("surge_multiplier", sa.Float, nullable=False),
        sa.Column("adjusted_fare", sa.Float, nullable=False),
        sa.Column("demand_level", sa.String(30), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_pricing_history_zone_id", "pricing_history", ["zone_id"])

    # ── eta_log ───────────────────────────────────────────────────
    op.create_table(
        "eta_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("pickup_zone", sa.Integer, nullable=False),
        sa.Column("dropoff_zone", sa.Integer, nullable=False),
        sa.Column("hour", sa.Integer, nullable=False),
        sa.Column("minute", sa.Integer, nullable=False),
        sa.Column("day_of_week", sa.Integer, nullable=False),
        sa.Column("base_eta_minutes", sa.Float, nullable=False),
        sa.Column("adjusted_eta_minutes", sa.Float, nullable=False),
        sa.Column("traffic_multiplier", sa.Float, nullable=False),
        sa.Column("distance_miles", sa.Float, nullable=False),
        sa.Column("graph_available", sa.Boolean, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_eta_log_created_at", "eta_log", ["created_at"])


def downgrade() -> None:
    op.drop_table("eta_log")
    op.drop_table("pricing_history")
    op.drop_table("predictions_log")
    op.drop_table("zone_stats")
    op.drop_table("zones")
