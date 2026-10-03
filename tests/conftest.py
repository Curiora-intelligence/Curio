import pytest_asyncio
from app.persistence.database import Database
from app.persistence.models import Base


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'curio.db'}")
    async with database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield database
    await database.close()
