from __future__ import annotations

import os
from sqlalchemy import create_engine, Engine
from dotenv import load_dotenv


load_dotenv()


def get_engine() -> Engine:
    required = {"POSTGRES_USER": None, "POSTGRES_PASSWORD": None, "POSTGRES_DB": None}
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in values."
        )

    user     = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5432")
    db       = os.getenv("POSTGRES_DB")
    ssl_mode = os.getenv("POSTGRES_SSL_MODE", "prefer")

    uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    # statement_timeout in ms — kills queries that hang beyond this limit
    statement_timeout_ms = int(os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS", "300000"))  # 5 min default
    connect_args: dict = {
        "sslmode": ssl_mode,
        "options": f"-c statement_timeout={statement_timeout_ms}",
    }

    return create_engine(
        uri,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
        connect_args=connect_args,
    )
