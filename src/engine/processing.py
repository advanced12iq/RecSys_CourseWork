import sys
from pathlib import Path

# Add the project root to the Python path (for quick testing)
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def process_data(last_date):
    # Convert last_date to pandas Timestamp for comparison
    last_date = pd.to_datetime(last_date)

    # Ensure the processed directory exists
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # --- Read Raw Data ---
    users_df = pd.read_csv(RAW_DATA_PATH / "users.csv")
    items_df = pd.read_csv(RAW_DATA_PATH / "items.csv")
    interactions_df = pd.read_csv(
        RAW_DATA_PATH / "interactions.csv",
        parse_dates=["last_watch_dt"],
        index_col=["user_id", "item_id"],
    )

    processed_path = PROCESSED_DATA_PATH / "interactions_processed.csv"
    old_interactions = (
        pd.read_csv(
            processed_path,
            parse_dates=["last_watch_dt"],
            index_col=["user_id", "item_id"],
        )
        if processed_path.exists()
        else pd.DataFrame()
    )
    processed_indices = old_interactions.index

    mask = ~interactions_df.index.isin(processed_indices) & (
        interactions_df["last_watch_dt"].apply(pd.to_datetime) < last_date
    )

    interactions_df = interactions_df[mask]

    # --- Users Preprocessing ---
    users_df["age"] = users_df["age"].fillna("age_unknown").astype("category")
    users_df["income"] = users_df["income"].fillna("income_unknown").astype("category")
    users_df["sex"] = (
        users_df["sex"]
        .fillna("sex_unknown")
        .replace({"М": "M", "Ж": "F"})
        .astype("category")
    )
    users_df["kids_flg"] = users_df["kids_flg"].astype("bool")

    # --- Items Preprocessing ---
    items_df["content_type"] = items_df["content_type"].astype("category")
    items_df["title"] = items_df["title"].str.lower()
    items_df["title_orig"] = items_df["title_orig"].fillna("None")
    items_df["release_year"] = items_df["release_year"].fillna(2020.0)
    items_df["release_year_cat"] = items_df["release_year"].copy()

    items_df.loc[items_df["release_year"] < 1920, "release_year_cat"] = "inf_1920"
    items_df.loc[items_df["release_year"] >= 2020, "release_year_cat"] = "2020_inf"

    for i in range(1920, 2020, 10):
        items_df.loc[
            (items_df["release_year"] >= i) & (items_df["release_year"] < i + 10),
            "release_year_cat",
        ] = f"{i}-{i + 10}"

    items_df = items_df.drop(columns=["release_year"])
    items_df["release_year_cat"] = items_df["release_year_cat"].astype("category")
    items_df["genres"] = items_df["genres"].astype("category")
    items_df["countries"] = (
        items_df["countries"]
        .fillna("Россия")
        .str.lower()
        .apply(lambda x: ", ".join(sorted(list(set(x.split(", "))))))
        .astype("category")
    )
    items_df["for_kids"] = items_df["for_kids"].fillna(0).astype("bool")
    items_df["age_rating"] = items_df["age_rating"].fillna(0).astype("category")
    items_df["studios"] = (
        items_df["studios"]
        .fillna("Unknown")
        .str.lower()
        .apply(lambda x: ", ".join(sorted(list(set(x.split(", "))))))
        .astype("category")
    )
    items_df["directors"] = (
        items_df["directors"].fillna("Unknown").str.lower().astype("category")
    )
    items_df["actors"] = items_df["actors"].fillna("Unknown").astype("category")
    items_df["keywords"] = items_df["keywords"].fillna("Unknown").astype("category")
    items_df["description"] = items_df["description"].fillna("-")

    # --- Interactions Preprocessing ---
    interactions_df["watched_pct"] = (
        interactions_df["watched_pct"].astype(pd.Int8Dtype()).fillna(0)
    )
    interactions_df["total_dur"] = interactions_df["total_dur"]
    interactions_df["last_watch_dt"] = pd.to_datetime(interactions_df["last_watch_dt"])

    # --- Save Processed Files ---
    users_df.to_csv(PROCESSED_DATA_PATH / "users_processed.csv", index=False)
    items_df.to_csv(PROCESSED_DATA_PATH / "items_processed.csv", index=False)
    interactions_df = pd.concat([old_interactions, interactions_df], axis=0)
    interactions_df.to_csv(
        PROCESSED_DATA_PATH / "interactions_processed.csv", index=True
    )

    # --- Return updated list of processed indices ---
    return interactions_df.index
