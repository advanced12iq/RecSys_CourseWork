from pathlib import Path

BASE_PATH = Path(__file__).parent.parent  # Go up two directories from current file
DATA_PATH = BASE_PATH / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH =DATA_PATH / "processed"