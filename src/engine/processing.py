import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from src.config import RAW_DATA_PATH, engine

# Add the project root to the Python path (for quick testing)
sys.path.append(str(Path(__file__).parent.parent))


def preprocess_users(df):
    df["age"] = df["age"].fillna("age_unknown").astype("category")
    df["income"] = df["income"].fillna("income_unknown").astype("category")
    df["sex"] = (
        df["sex"]
        .fillna("sex_unknown")
        .replace({"Ðœ": "M", "Ð–": "F"})
        .astype("category")
    )
    df["kids_flg"] = df["kids_flg"].astype("bool")
    return df


def preprocess_items(df):
    df["content_type"] = df["content_type"].astype("category")
    df["release_year_cat"] = df["release_year"].astype("category")
    df["genres"] = df["genres"].astype("category")
    df["countries"] = df["countries"].astype("category")
    df["for_kids"] = df["for_kids"].astype("bool")
    df["age_rating"] = df["age_rating"].fillna(0).astype("category")
    df["studios"] = df["studios"].astype("category")
    df["directors"] = df["directors"].astype("category")
    df["actors"] = df["actors"].astype("category")
    df["keywords"] = df["keywords"].astype("category")
    df = df.drop(columns=["release_year"])
    return df


def preprocess_interactions(df):
    df["watched_pct"] = df["watched_pct"].fillna(0).astype("int8")
    df["last_watch_dt"] = pd.to_datetime(df["last_watch_dt"])

    return df


@lru_cache(maxsize=2)
def load_data():
    """Cached version of data loader."""
    users_df = pd.read_sql("SELECT * FROM users_processed", engine)
    items_df = pd.read_sql("SELECT * FROM items_processed", engine)
    interactions_df = pd.read_sql("SELECT * FROM interactions_processed", engine)

    return (users_df, items_df, interactions_df)


# Invalidate cache when data changes
def invalidate_caches():
    load_data.cache_clear()


def process_data(last_date):
    last_date = pd.to_datetime(last_date)

    # Load raw data
    users_df = pd.read_csv("data/raw/users.csv")
    items_df = pd.read_csv("data/raw/items.csv")
    interactions_df = pd.read_csv(
        "data/raw/interactions.csv",
        parse_dates=["last_watch_dt"],
        index_col=["user_id", "item_id"],
    )

    # Load existing interactions
    try:
        old_interactions = pd.read_sql(
            "SELECT user_id, item_id FROM interactions_processed", engine
        ).set_index(["user_id", "item_id"])
    except Exception:
        old_interactions = pd.DataFrame().set_index(["user_id", "item_id"])

    # Filter new data
    mask = ~interactions_df.index.isin(old_interactions.index) & (
        interactions_df["last_watch_dt"] < last_date
    )
    interactions_df = interactions_df[mask]

    # Preprocess
    users_df = preprocess_users(users_df)
    items_df = preprocess_items(items_df)
    interactions_df = preprocess_interactions(interactions_df)

    # Truncate and update tables
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE users_processed"))
        conn.execute(text("TRUNCATE TABLE items_processed"))

    users_df.to_sql("users_processed", engine, if_exists="append", index=False)
    items_df.to_sql("items_processed", engine, if_exists="append", index=False)
    interactions_df.to_sql(
        "interactions_processed", engine, if_exists="append", index=True
    )

    # Invalidate cached data
    invalidate_caches()

    return interactions_df.index
