from datetime import timedelta
from itertools import cycle, islice

import pandas as pd
from implicit.nearest_neighbours import TFIDFRecommender
from src.engine.utils import compute_metrics, get_coo_matrix, setup_time_range_split


from functools import lru_cache

@lru_cache(maxsize=128)
def generate_implicit_recs_mapper_cached(
    user_id, model, train_matrix, top_N, item_inv_mapping, filter_already_liked_items
):
    recs = model.recommend(
        user_id,
        train_matrix,
        N=top_N,
        filter_already_liked_items=filter_already_liked_items
    )
    return [item_inv_mapping[item] for item, _ in recs]


class PopularRecommender:
    def __init__(self, max_K=10, days=30, item_column="item_id", dt_column="last_watch_dt"):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []

    def fit(
        self,
        df,
    ):
        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = (
            df.loc[df[self.dt_column] > min_date, self.item_column]
            .value_counts()
            .head(self.max_K)
            .index.values
        )

    def recommend(self, users=None, N=10):
        recs = self.recommendations[:N]
        if users is None:
            return recs
        else:
            return list(islice(cycle([recs]), len(users)))


def get_top_recommendations(model, train_data, days=7, date_col="last_watch_dt"):
    """Generate top N recommendations using a model."""
    model.days = days
    model.fit(train_data)
    return model.recommend()


def generate_user_recommendations(model, test_users, top_N=10):
    """Generate recommendations for test users with ranking."""
    recs = pd.DataFrame({"user_id": test_users})
    recs["item_id"] = model.recommend(recs['user_id'], N=top_N)
    recs = recs.explode("item_id")
    recs["rank"] = recs.groupby("user_id").cumcount() + 1
    return recs


def validate_model(
    model_class,
    interactions_df,
    users_df,
    items_df,
    top_N=10,
    last_n_days=7,
    date_col="last_watch_dt",
):
    """Run cross-validation for a recommendation model."""
    cv_splits = setup_time_range_split(interactions_df, last_date_col=date_col)
    all_metrics = []  # Use a list to collect metrics

    for train_idx, test_idx, _ in cv_splits:
        train = interactions_df.loc[train_idx]
        test = interactions_df.loc[test_idx]

        # Train model
        model = model_class(days=last_n_days, dt_column=date_col)
        model.fit(train)

        # Generate recommendations
        recs = generate_user_recommendations(model, test["user_id"].unique(), top_N)

        # Calculate metrics
        metrics = compute_metrics(train, test, recs, top_N)
        all_metrics.append(metrics)  # Append the metrics DataFrame to the list

    # Concatenate all metric DataFrames at once
    results = pd.DataFrame(all_metrics)
    return results


def generate_socdem_recommendations(train_data, users_df, last_n_days=7):
    """Generate recommendations based on socio-demographic groups."""
    date_window = train_data["last_watch_dt"].max() - timedelta(days=last_n_days)
    train_slice = pd.merge(
        train_data[train_data["last_watch_dt"] >= date_window],
        users_df,
        on="user_id",
        how="left",
    ).fillna(
        {
            "age": "age_unknown",
            "sex": "sex_unknown",
            "income": "income_unknown",
            "kids_flg": False,
        }
    )

    soc_dem_rec = (
        train_slice.groupby(["age", "sex", "income", "item_id"])
        .size()
        .reset_index(name="count")
    )

    top_soc_dem = []
    for age in soc_dem_rec["age"].unique():
        for income in soc_dem_rec["income"].unique():
            for sex in soc_dem_rec["sex"].unique():
                group_rec = (
                    soc_dem_rec[
                        (soc_dem_rec["age"] == age)
                        & (soc_dem_rec["income"] == income)
                        & (soc_dem_rec["sex"] == sex)
                    ]
                    .sort_values("count", ascending=False)
                    .head(10)
                )
                top_soc_dem.append([age, income, sex, group_rec["item_id"].values])

    return pd.DataFrame(top_soc_dem, columns=["age", "income", "sex", "item_id"])


def validate_socdem_model(interactions_df, users_df, top_N=10, last_n_days=7):
    """Run cross-validation for socio-demographic model."""
    cv_splits = setup_time_range_split(interactions_df)
    all_metrics = []

    for train_idx, test_idx, _ in cv_splits:
        train = interactions_df.loc[train_idx]
        test = interactions_df.loc[test_idx]

        # Generate group recommendations
        socdem_recs = generate_socdem_recommendations(train, users_df, last_n_days)

        # Merge with test users
        recs = pd.DataFrame({"user_id": test["user_id"].unique()})
        recs = pd.merge(
            recs,
            users_df[["user_id", "age", "sex", "income"]],
            on="user_id",
            how="left",
        ).fillna(
            {"age": "age_unknown", "sex": "sex_unknown", "income": "income_unknown"}
        )

        recs = pd.merge(recs, socdem_recs, on=["age", "sex", "income"], how="left")
        recs = recs.drop(columns=["age", "sex", "income"]).explode("item_id")
        recs["rank"] = recs.groupby("user_id").cumcount() + 1

        # Calculate metrics
        metrics = compute_metrics(train, test, recs, top_N)
        all_metrics.append(metrics)
    
    results = pd.DataFrame(all_metrics)
    return results


def validate_tfidf_model(interactions_df, top_N=10, window_days=60):
    """Run cross-validation for TF-IDF model."""
    cv_splits = setup_time_range_split(interactions_df)
    all_metrics = []

    # Create user/item mappings
    users_inv_mapping = dict(enumerate(interactions_df["user_id"].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    items_inv_mapping = dict(enumerate(interactions_df["item_id"].unique()))
    items_mapping = {v: k for k, v in items_inv_mapping.items()}

    for train_idx, test_idx, _ in cv_splits:
        train = interactions_df.loc[train_idx]
        date_window = train["last_watch_dt"].max() - timedelta(days=window_days)
        train_window = train[train["last_watch_dt"] >= date_window]
        test_window = interactions_df.loc[test_idx]

        # Create matrix and train model
        train_mat = get_coo_matrix(
            train_window, users_mapping=users_mapping, items_mapping=items_mapping
        ).tocsr()

        model = TFIDFRecommender(K=top_N)
        model.fit(train_mat.T)

        # Generate recommendations
        mapper = generate_implicit_recs_mapper_cached(
            model, train_mat, top_N, users_mapping, items_inv_mapping, True
        )
        recs = pd.DataFrame({"user_id": test_window["user_id"].unique()})
        recs["item_id"] = recs["user_id"].map(mapper)
        recs = recs.explode("item_id")
        recs["rank"] = recs.groupby("user_id").cumcount() + 1

        # Calculate metrics
        metrics = compute_metrics(train_window, test_window, recs, top_N)
        all_metrics.append(metrics)

    results = pd.DataFrame(all_metrics)
    return results
