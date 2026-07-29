import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_PATH

def add_id_column_to_interactions():
    # Define the file path
    interactions_path = RAW_DATA_PATH / "interactions.csv"

    # Read the raw interactions data
    interactions_df = pd.read_csv(interactions_path, parse_dates=["last_watch_dt"], index_col=['user_id', 'item_id'])

    # Add a new 'id' column with unique row numbers (starting from 0)
    # interactions_df = interactions_df.drop(['id'], axis=1)
    interactions_df = interactions_df.drop(['Unnamed: 0'], axis=1)

    # Save the updated DataFrame back to the same file (or a new one)
    interactions_df.to_csv(interactions_path, index=True)

    print("✅ 'id' column added and saved to", interactions_path)

if __name__ == "__main__":
    add_id_column_to_interactions()