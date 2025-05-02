import os
from pathlib import Path
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

BASE_PATH = Path(__file__).parent.parent  # Go up two directories from current file
DATA_PATH = BASE_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

CONNECTION_STRING = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(
    CONNECTION_STRING,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Ensure indexes exist
def create_indexes():
    with engine.begin() as conn:
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_interactions_user_id') THEN
                    CREATE INDEX idx_interactions_user_id ON interactions_processed(user_id);
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_interactions_item_id') THEN
                    CREATE INDEX idx_interactions_item_id ON interactions_processed(item_id);
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_interactions_date') THEN
                    CREATE INDEX idx_interactions_date ON interactions_processed(last_watch_dt);
                END IF;
            END $$;
        """))
