import numpy as np
import scipy.sparse as sp
import pandas as pd
from more_itertools import pairwise
from src.config import PROCESSED_DATA_PATH
import datetime


def calculate_novelty(train_interactions, recommendations, top_n):
    users = recommendations["user_id"].unique()
    n_users = train_interactions["user_id"].nunique()
    n_users_per_item = train_interactions.groupby("item_id")["user_id"].nunique()
    recommendations = recommendations.loc[recommendations["rank"] <= top_n].copy()
    recommendations["n_users_per_item"] = recommendations["item_id"].map(
        n_users_per_item
    )
    recommendations["n_users_per_item"] = recommendations["n_users_per_item"].fillna(1)
    recommendations["item_novelty"] = -np.log2(
        recommendations["n_users_per_item"] / n_users
    )
    item_novelties = recommendations[["user_id", "rank", "item_novelty"]]
    miuf_at_k = item_novelties.loc[
        item_novelties["rank"] <= top_n, ["user_id", "item_novelty"]
    ]
    miuf_at_k = miuf_at_k.groupby("user_id").agg("mean").squeeze()
    return miuf_at_k.reindex(users).mean()


def compute_metrics(train, test, recs, top_N):
    result = {}
    test_recs = test.set_index(["user_id", "item_id"]).join(
        recs.set_index(["user_id", "item_id"])
    )
    test_recs = test_recs.sort_values(by=["user_id", "rank"])
    test_recs["users_item_count"] = test_recs.groupby(level="user_id")[
        "rank"
    ].transform(np.size)
    test_recs["reciprocal_rank"] = (1 / test_recs["rank"]).fillna(0)
    test_recs["cumulative_rank"] = test_recs.groupby(level="user_id").cumcount() + 1
    test_recs["cumulative_rank"] = test_recs["cumulative_rank"] / test_recs["rank"]
    users_count = test_recs.index.get_level_values("user_id").nunique()
    result[f"MAP@{top_N}"] = (
        test_recs["cumulative_rank"] / test_recs["users_item_count"]
    ).sum() / users_count
    result[f"Novelty@{top_N}"] = calculate_novelty(train, recs, top_N)
    return pd.Series(result)


class TimeRangeSplit:
    def __init__(
        self,
        start_date,
        end_date=None,
        freq="D",
        periods=None,
        tz=None,
        normalize=False,
        closed=None,
        train_min_date=None,
        filter_cold_users=True,
        filter_cold_items=True,
        filter_already_seen=True,
    ):
        self.start_date = start_date
        if end_date is None and periods is None:
            raise ValueError(
                'Either "end_date" or "periods" must be non-zero, not both at the same time.'
            )
        self.end_date = end_date
        self.freq = freq
        self.periods = periods
        self.tz = tz
        self.normalize = normalize
        self.closed = closed
        self.train_min_date = pd.to_datetime(train_min_date, errors="raise")
        self.filter_cold_users = filter_cold_users
        self.filter_cold_items = filter_cold_items
        self.filter_already_seen = filter_already_seen
        self.date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq,
            periods=periods,
            tz=tz,
            normalize=normalize,
        )
        self.max_n_splits = max(0, len(self.date_range) - 1)
        if self.max_n_splits == 0:
            raise ValueError("Provided parametrs set an empty date range.")

    def split(
        self,
        df,
        user_column="user_id",
        item_column="item_id",
        datetime_column="date",
        fold_stats=False,
    ):
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            train_min_mask = df_datetime >= self.train_min_date
        else:
            train_min_mask = df_datetime.notnull()
        date_range = self.date_range[
            (self.date_range >= df_datetime.min())
            & (self.date_range <= df_datetime.max())
        ]
        for start, end in pairwise(date_range):
            fold_info = {"Start date": start, "End date": end}
            train_mask = train_min_mask & (df_datetime < start)
            train_idx = df.index[train_mask]
            if fold_stats:
                fold_info["Train"] = len(train_idx)
            test_mask = (df_datetime >= start) & (df_datetime < end)
            test_idx = df.index[test_mask]
            if self.filter_cold_users:
                new = np.setdiff1d(
                    df.loc[test_idx, user_column].unique(),
                    df.loc[train_idx, user_column].unique(),
                )
                new_idx = df.index[test_mask & df[user_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info["New users"] = len(new)
                    fold_info["New users interactions"] = len(new_idx)
            if self.filter_cold_items:
                new = np.setdiff1d(
                    df.loc[test_idx, item_column].unique(),
                    df.loc[train_idx, item_column].unique(),
                )
                new_idx = df.index[test_mask & df[item_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info["New items"] = len(new)
                    fold_info["New items interactions"] = len(new_idx)
            if self.filter_already_seen:
                user_item = [user_column, item_column]
                train_pairs = df.loc[train_idx, user_item].set_index(user_item).index
                test_pairs = df.loc[test_idx, user_item].set_index(user_item).index
                intersection = train_pairs.intersection(test_pairs)
                print(f"Already seen number: {len(intersection)}")
                test_idx = test_idx[~test_pairs.isin(intersection)]
                if fold_stats:
                    fold_info["Known interactions"] = len(intersection)
            if fold_stats:
                fold_info["Test"] = len(test_idx)
            yield (train_idx, test_idx, fold_info)

    def get_n_splits(self, df, datetime_column="date"):
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            df_datetime = df_datetime[df_datetime >= self.train_min_date]
        date_range = self.date_range[
            (self.date_range >= df_datetime.min())
            & (self.date_range <= df_datetime.max())
        ]
        return max(0, len(date_range) - 1)


def get_coo_matrix(
    df,
    user_col="user_id",
    item_col="item_id",
    weight_col=None,
    users_mapping={},
    items_mapping={},
):
    if weight_col is None:
        weights = np.ones(len(df), dtype=np.float32)
    else:
        weights = df[weight_col].astype(np.float32)
    interaction_matrix = sp.coo_matrix(
        (
            weights,
            (df[user_col].map(users_mapping.get), df[item_col].map(items_mapping.get)),
        )
    )
    return interaction_matrix


def split_train_test(interactions_df, date_col="last_watch_dt"):
    """Split data into train and test based on latest date."""
    max_date = interactions_df[date_col].max()
    test = interactions_df[interactions_df[date_col] == max_date]
    train = interactions_df[interactions_df[date_col] < max_date]
    return train, test


def generate_submission(
    model, interactions_df, submission_path, users_df, items_df, top_N=10, last_n_days=7
):
    """Generate submission file with recommendations."""
    model.days = last_n_days
    model.fit(interactions_df)

    submission = pd.read_csv(submission_path)
    recs = pd.DataFrame({"user_id": submission["user_id"].unique()})
    recs["item_id"] = recs["user_id"].map(lambda x: model.recommend([x], N=top_N))
    recs = recs.explode("item_id")
    recs["rank"] = recs.groupby("user_id").cumcount() + 1
    recs = recs.groupby("user_id").agg({"item_id": list}).reset_index()
    recs.to_csv(submission_path, index=False)
    return recs


def setup_time_range_split(data, last_date_col="last_watch_dt", folds=3):
    """Configure time-based cross-validation."""
    last_date = data[last_date_col].max().normalize()
    start_date = last_date - datetime.timedelta(days=folds * 7)
    cv = TimeRangeSplit(start_date=start_date, periods=folds + 1, freq="W")
    return cv.split(data, fold_stats=True, datetime_column=last_date_col)


def map_item_titles(items_df):
    """Map item IDs to titles."""
    return pd.Series(items_df["title"].values, index=items_df["item_id"]).to_dict()
