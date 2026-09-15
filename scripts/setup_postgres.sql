-- UrbanFlow PostgreSQL setup (run once as superuser in pgAdmin Query Tool)
--
-- After running, set in .env:
--   DATABASE_URL=postgresql+asyncpg://urbanflow:urbanflow@localhost:5432/urbanflow
--
-- Alternative: use Docker instead (docker compose up db -d) with postgres/postgres

CREATE ROLE urbanflow LOGIN PASSWORD 'urbanflow';

CREATE DATABASE urbanflow OWNER urbanflow;

GRANT ALL PRIVILEGES ON DATABASE urbanflow TO urbanflow;
