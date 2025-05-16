from datetime import timedelta
from itertools import cycle, islice

import pandas as pd
from implicit.nearest_neighbours import TFIDFRecommender
from src.engine.utils import compute_metrics, get_coo_matrix, setup_time_range_split

from catboost import CatBoostRegressor
from src.engine.processing import load_data


from functools import lru_cache
import faiss
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


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

class ContentBasedFaissRecommender:
    def __init__(self, days=7, dt_column='last_watch_dt', top_n=10):
        self.days = days
        self.dt_column = dt_column
        self.top_n = top_n
        self.item_ids = []
        self.item_vectors = None
        self.user_interactions = {}
        self.vectorizer = None
        self.index = None

    def fit(self, interactions_df):
        """
        Train the recommender by building a Faiss index of item vectors.
        """
        # Load full data
        _, items_df, _ = load_data()

        # Step 1: Create item feature vectors using TF-IDF
        self._create_item_vectors(items_df)

        # Step 2: Build Faiss index
        self._build_index()

        # Step 3: Map users to their recent interactions
        self._map_user_interactions(interactions_df)

    def _create_item_vectors(self, items_df):
        """
        Generate item embeddings using TF-IDF on combined text features.
        """
        # Combine relevant features (can include more like directors, actors, etc.)
        items_df['combined_features'] = (
            items_df['genres'].fillna('') + ' ' + items_df['keywords'].fillna('')
        )

        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = self.vectorizer.fit_transform(items_df['combined_features'])

        # Convert sparse matrix to dense and normalize for cosine similarity
        item_features = tfidf_matrix.toarray().astype('float32')
        item_features = normalize(item_features, axis=1)

        # Store vectors and corresponding item IDs
        self.item_vectors = item_features
        self.item_ids = items_df['item_id'].tolist()

    def _build_index(self):
        """
        Build Faiss index for efficient similarity search.
        """
        dimension = self.item_vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity
        self.index.add(self.item_vectors)

    def _map_user_interactions(self, interactions_df):
        """
        Map each user to their recently interacted item IDs (within time window).
        """
        min_date = interactions_df[self.dt_column].max() - pd.Timedelta(days=self.days)
        recent_interactions = interactions_df[interactions_df[self.dt_column] > min_date]
        self.user_interactions = recent_interactions.groupby('user_id')['item_id'].apply(list).to_dict()

    def recommend(self, user_ids, N=None):
        """
        Generate top-N recommendations for each user.
        """
        if N is None:
            N = self.top_n

        recommendations = []

        for user_id in user_ids:
            if user_id not in self.user_interactions:
                recommendations.append([])
                continue

            interacted_items = self.user_interactions[user_id]
            interacted_indices = [
                idx for idx, item_id in enumerate(self.item_ids) if item_id in interacted_items
            ]

            if not interacted_indices:
                recommendations.append([])
                continue

            # Compute user profile: average of interacted item vectors
            user_profile = np.mean(self.item_vectors[interacted_indices], axis=0).reshape(1, -1)

            # Search Faiss index for top candidates
            _, indices = self.index.search(user_profile, N + len(interacted_items))
            candidate_ids = [self.item_ids[i] for i in indices[0]]

            # Exclude already interacted items and return top-N
            recs = [iid for iid in candidate_ids if iid not in interacted_items][:N]
            recommendations.append(recs)

        return recommendations

    def score_items(self, user_id, item_ids):
        """
        Score a list of items for a user using content similarity.
        """
        if user_id not in self.user_interactions:
            return pd.DataFrame({'item_id': item_ids, 'content_score': 0})
        
        interacted_items = self.user_interactions[user_id]
        interacted_indices = [i for i, iid in enumerate(self.item_ids) if iid in interacted_items]
        if not interacted_indices:
            return pd.DataFrame({'item_id': item_ids, 'content_score': 0})

        user_profile = np.mean(self.item_vectors[interacted_indices], axis=0).reshape(1, -1)

        # Get vectors for candidate items
        item_indices = [self.item_ids.index(iid) for iid in item_ids if iid in self.item_ids]
        if not item_indices:
            return pd.DataFrame({'item_id': item_ids, 'content_score': 0})

        item_vectors = self.item_vectors[item_indices]
        scores = user_profile @ item_vectors.T  # Cosine similarity
        score_map = {self.item_ids[i]: float(scores[0][idx]) for idx, i in enumerate(item_indices)}

        return pd.DataFrame({
            'item_id': item_ids,
            'content_score': [score_map.get(iid, 0) for iid in item_ids]
        })


class FactorizationMachineRecommender:
    def __init__(self, days=7, dt_column='last_watch_dt', top_n=10, model_params=None):
        self.days = days
        self.dt_column = dt_column
        self.top_n = top_n
        self.model_params = model_params or {}
        self.model = CatBoostRegressor(**self.model_params)
        self.feature_columns = []
        self.categorical_features = []

    def fit(self, interactions_df):
        # Load full user and item data
        users_df, items_df, _ = load_data()

        users_df = users_df.sample(500)

        # Prefix user and item columns to distinguish them
        users_prefixed = users_df.add_prefix('user_')
        items_prefixed = items_df.add_prefix('item_')

        # Rename ID columns back to original
        users_prefixed = users_prefixed.rename(columns={'user_user_id': 'user_id'})
        items_prefixed = items_prefixed.rename(columns={'item_item_id': 'item_id'})

        # Merge interactions with users and items
        merged = interactions_df.merge(users_prefixed, on='user_id', how='left') \
                                .merge(items_prefixed, on='item_id', how='left')

        # Filter by date window
        min_date = merged[self.dt_column].max() - pd.Timedelta(days=self.days)
        train_data = merged[merged[self.dt_column] > min_date]

        # Define target and features
        y = train_data['watched_pct']

        # Drop irrelevant columns
        feature_columns = train_data.columns.tolist()
        feature_columns.remove('watched_pct')
        feature_columns.remove('user_id')
        feature_columns.remove('item_id')
        feature_columns.remove(self.dt_column)

        X = train_data[feature_columns]

        # Identify categorical features
        self.categorical_features = X.select_dtypes(include='category').columns.tolist()
        self.feature_columns = feature_columns

        # Train the model
        self.model.fit(X, y, cat_features=self.categorical_features)

    def recommend(self, user_ids, N=None):
        if N is None:
            N = self.top_n

        recommendations = []

        # Load full data
        users_df, items_df, interactions_df = load_data()

        # Prefix user and item columns
        users_prefixed = users_df.add_prefix('user_')
        users_prefixed = users_prefixed.rename(columns={'user_user_id': 'user_id'})
        items_prefixed = items_df.add_prefix('item_')
        items_prefixed = items_prefixed.rename(columns={'item_item_id': 'item_id'})

        for user_id in user_ids:
            # Get user features
            user_row = users_prefixed[users_prefixed['user_id'] == user_id]
            if user_row.empty:
                recommendations.append([])
                continue

            # Repeat user row for all items
            user_items = pd.concat([user_row] * len(items_prefixed), ignore_index=True).join(
                items_prefixed.reset_index(drop=True), how='inner'
            )

            # Predict scores
            scores = self.model.predict(user_items[self.feature_columns])

            # Add scores to dataframe
            user_items['score'] = scores

            # Get interacted items
            interacted = interactions_df[interactions_df['user_id'] == user_id]['item_id']

            # Filter out interacted items and sort
            recs = user_items[~user_items['item_item_id'].isin(interacted)] \
                .sort_values('score', ascending=False) \
                .head(N)

            recommendations.append(recs['item_item_id'].tolist())

        return recommendations
    
    # src/engine/methods.py

    def score_items(self, user_id, item_ids):
        """
        Score a list of items for a user using factorization machine model.
        """
        users_df, items_df, _ = load_data()

        # Prefix columns
        users_prefixed = users_df.add_prefix('user_').rename(columns={'user_user_id': 'user_id'})
        items_prefixed = items_df.add_prefix('item_').rename(columns={'item_item_id': 'item_id'})

        # Get user row
        user_row = users_prefixed[users_prefixed['user_id'] == user_id]
        if user_row.empty:
            return pd.DataFrame({'item_id': item_ids, 'fm_score': 0})

        # Repeat user for each item
        user_items = pd.concat([user_row] * len(item_ids), ignore_index=True)
        item_rows = items_prefixed[items_prefixed['item_item_id'].isin(item_ids)]
        user_items = user_items.join(item_rows.reset_index(drop=True), how='inner')

        # Predict scores
        scores = self.model.predict(user_items[self.feature_columns])
        user_items['fm_score'] = scores

        return pd.DataFrame({
            'item_id': user_items['item_item_id'],
            'fm_score': user_items['fm_score']
        })


# src/engine/methods.py

class HybridRecommender:
    def __init__(self, base_recommenders, content_recommender, fm_recommender, top_n=100, final_top_n=10):
        self.base_recommenders = base_recommenders
        self.content_recommender = content_recommender
        self.fm_recommender = fm_recommender
        self.top_n = top_n
        self.final_top_n = final_top_n
        self.item_ids = []

    def fit(self, interactions_df):
        """Fit all internal models"""
        for rec in self.base_recommenders:
            rec.fit(interactions_df)
        self.content_recommender.fit(interactions_df)
        self.fm_recommender.fit(interactions_df)

    def recommend(self, user_ids, N=None):
        if N is None:
            N = self.final_top_n

        results = []

        for user_id in user_ids:
            # Stage 1: Get top 100 items from base recommenders
            top_items = set()
            for rec in self.base_recommenders:
                recs = rec.recommend([user_id], N=self.top_n)[0]
                top_items.update(recs)
            top_items = list(top_items)

            if not top_items:
                results.append([])
                continue

            # Stage 2: Score items with advanced models
            content_scores = self.content_recommender.score_items(user_id, top_items)
            fm_scores = self.fm_recommender.score_items(user_id, top_items)

            # Merge scores
            scores = content_scores.merge(fm_scores, on='item_id', how='outer').fillna(0)

            # Normalize scores
            scores['content_score'] = scores['content_score'] / (scores['content_score'].max() or 1)
            scores['fm_score'] = scores['fm_score'] / (scores['fm_score'].max() or 1)

            # Combine scores (e.g., average)
            scores['combined_score'] = 0.5 * scores['content_score'] + 0.5 * scores['fm_score']

            # Rerank and return top-N
            reranked = scores.sort_values('combined_score', ascending=False).head(N)
            results.append(reranked['item_id'].tolist())

        return results


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
