from src.engine.processing import process_data, load_data
from src.engine.methods import validate_model, PopularRecommender, validate_tfidf_model, validate_socdem_model

from dotenv import load_dotenv

load_dotenv()

def generate_recs():
    last_date = "2023-01-01"
    processed_indices = process_data(last_date)
    users_df, items_df, interactions_df = load_data()
    results = validate_model(PopularRecommender, interactions_df, users_df, items_df)
    # print(results)
    # results = validate_tfidf_model(interactions_df, users_df)
    print(results)


def main():
    generate_recs()

main()