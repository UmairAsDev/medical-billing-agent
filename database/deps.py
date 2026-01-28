import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from contextlib import contextmanager, asynccontextmanager
from database.db import SessionLocal, AsyncSessionLocal

@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()




@asynccontextmanager
async def async_db_session():
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()

