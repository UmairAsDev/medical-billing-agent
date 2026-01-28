import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from config.gunicorn_conf import settings
from loguru import logger




sync_url = URL.create(
    "mysql+pymysql",
    username=settings.DB_USERNAME,
    password=settings.DB_PASSWORD,
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    database=settings.DB_NAME,
)
logger.success(f"connection to database: {sync_url}")


async_url = URL.create(
    "mysql+aiomysql",
    username=settings.DB_USERNAME,
    password=settings.DB_PASSWORD,
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    database=settings.DB_NAME,
)
    


engine = create_engine(sync_url, echo=True, pool_pre_ping=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

async_engine = create_async_engine(async_url, pool_pre_ping=True, echo=True)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    autoflush=False,
    autocommit=False,
)

class Base(DeclarativeBase):
    pass


