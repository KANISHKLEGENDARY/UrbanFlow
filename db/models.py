# This code is responsible for bridging the predictions between python and postgreSQL. SQLAlchemy was used to to directly store the predictions and other details without writing sql queries. All the tables are mentioned in this code and they have functionality and need according to their significance. There are 6 tables:- Zone, ZoneStat, PredictionLog, PricingHistory, ETALog, User. Each table have different types of functionalities:- 
# Zone table:- This table stores the information about each zone, such as the zone id, zone name, borough, service zone, latitude and longitude. 
# ZoneStat table:- This table stores the statistics of each zone, such as the zone id, time slot, zone demand mean, zone demand std, zone demand max, zone slot demand, borough demand mean, raw demand max. 
# PredictionLog table:- This table stores the predictions of each zone, such as the zone id, time slot, predicted demand, demand raw, surge multiplier, demand level. 
# PricingHistory table:- This table stores the pricing information of each zone, such as the zone id, time slot, base fare, surge multiplier, adjusted fare, demand level. 
# ETALog table:- This table stores the eta information of each zone, such as the zone id, time slot, base eta, adjusted eta, traffic multiplier, distance miles, graph available. 
# User table:- This table stores the user information, such as the user id, username, email, hashed password, is active, is admin, created at, updated at.

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Boolean, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Zone(Base):
    __tablename__ = "zones"

    zone_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone_name: Mapped[str] = mapped_column(String(160), nullable=False)
    borough: Mapped[str] = mapped_column(String(80), nullable=False)
    service_zone: Mapped[str | None] = mapped_column(String(80), nullable=True)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lng: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class ZoneStat(Base):
    __tablename__ = "zone_stats"

    zone_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    time_slot: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone_demand_mean: Mapped[float] = mapped_column(Float, nullable=False)
    zone_demand_std: Mapped[float] = mapped_column(Float, nullable=False)
    zone_demand_max: Mapped[float] = mapped_column(Float, nullable=False)
    zone_slot_demand: Mapped[float] = mapped_column(Float, nullable=False)
    borough_demand_mean: Mapped[float] = mapped_column(Float, nullable=False)
    raw_demand_max: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_type: Mapped[str] = mapped_column(String(40), nullable=False)
    zone_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    minute: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_demand: Mapped[float] = mapped_column(Float, nullable=False)
    demand_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    surge_multiplier: Mapped[float | None] = mapped_column(Float, nullable=True)
    demand_level: Mapped[str | None] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class PricingHistory(Base):
    __tablename__ = "pricing_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    zone_id: Mapped[int] = mapped_column(Integer, nullable=False)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    minute: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    base_fare: Mapped[float] = mapped_column(Float, nullable=False)
    surge_multiplier: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_fare: Mapped[float] = mapped_column(Float, nullable=False)
    demand_level: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class ETALog(Base):
    __tablename__ = "eta_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pickup_zone: Mapped[int] = mapped_column(Integer, nullable=False)
    dropoff_zone: Mapped[int] = mapped_column(Integer, nullable=False)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    minute: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    base_eta_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_eta_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    traffic_multiplier: Mapped[float] = mapped_column(Float, nullable=False)
    distance_miles: Mapped[float] = mapped_column(Float, nullable=False)
    graph_available: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

