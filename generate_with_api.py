from flask import Flask, request, jsonify
import logging
import pandas as pd
from src.engine.methods import PopularRecommender, validate_model, generate_socdem_recommendations
from src.engine.processing import load_data
import os

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Global variables to store initialized models
popular_model = None
socdem_recs = None

def initialize_models():
    """Initialize recommendation models at startup"""
    global popular_model, socdem_recs
    
    # Load data
    users_df, items_df, interactions_df = load_data()
    
    # Initialize Popular Recommender
    popular_model = PopularRecommender(days=7)
    popular_model.fit(interactions_df)
    
    # Initialize Socio-demographic Recommender
    socdem_recs = generate_socdem_recommendations(interactions_df, users_df, last_n_days=7)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Unified recommendation endpoint (supports 'popular' and 'socdem' methods)"""
    user_id = request.args.get('user_id')
    method = request.args.get('method', 'popular')
    top_n = int(request.args.get('top_n', 10))
    
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
        
    try:
        if method == 'popular':
            # Use popular recommender
            recs = popular_model.recommend([int(user_id)], N=top_n)
            result = [{'item_id': str(item)} for item in recs[0]]
            
        elif method == 'socdem':
            # Get user data
            users_df, _, _ = load_data()
            user_data = users_df[users_df['user_id'] == int(user_id)]
            
            if user_data.empty:
                return jsonify({'error': 'User not found'}), 404
                
            # Get user demographics
            user_row = user_data.iloc[0]
            age = user_row['age'] or 'age_unknown'
            income = user_row['income'] or 'income_unknown'
            sex = user_row['sex'] or 'sex_unknown'
            
            # Filter recommendations by demographics
            filtered = socdem_recs[
                (socdem_recs['age'] == age) &
                (socdem_recs['income'] == income) &
                (socdem_recs['sex'] == sex)
            ]
            
            if filtered.empty:
                # Fallback to popular recommendations if no demographic match
                recs = popular_model.recommend([int(user_id)], N=top_n)
                result = [{'item_id': str(item)} for item in recs[0]]
            else:
                result = [{'item_id': str(item)} for item in filtered.iloc[0]['item_id']]
                
        else:
            return jsonify({
                'error': f'Unknown method: {method}. Available methods: "popular", "socdem"'
            }), 400
            
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    initialize_models()
    app.run(host='0.0.0.0', port=5000, debug=True)