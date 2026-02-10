from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import pickle
from predict_price import PricePredictor
from datetime import datetime

app = Flask(__name__)

# Global variables to store predictor and data
predictor = None
df = None
MODELS_DIR = "saved_models"

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ==================== Helper Functions ====================

def save_model(model_name, predictor_obj):
    """Save a trained model to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)

    with open(filepath, 'wb') as f:
        pickle.dump({
            'model': predictor_obj.trained_models[model_name],
            'rmse': predictor_obj.rmse_scores[model_name],
            'X_columns': list(predictor_obj.X.columns)
        }, f)

    return filename

def load_model(filename):
    """Load a trained model from disk"""
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# ==================== Web Routes ====================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Get current status of the application"""
    global predictor, df

    return jsonify({
        'data_loaded': df is not None,
        'predictor_initialized': predictor is not None,
        'trained_models': list(predictor.trained_models.keys()) if predictor else [],
        'dataset_shape': df.shape if df is not None else None,
        'available_models': ['LinearRegression', 'RandomForest', 'XGBoost']
    })

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load dataset from file"""
    global df, predictor

    try:
        data = request.get_json()
        filepath = data.get('filepath', 'clean_dataset.csv')
        target = data.get('target', 'buy_price')

        # Load the dataset
        df = pd.read_csv(filepath)

        # Initialize predictor
        predictor = PricePredictor(df, target=target)

        return jsonify({
            'success': True,
            'message': 'Data loaded successfully',
            'shape': df.shape,
            'columns': list(df.columns),
            'target': target
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train one or more models"""
    global predictor

    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'No data loaded. Please load data first.'
        }), 400

    try:
        data = request.get_json()
        model_names = data.get('models', None)  # None trains all models

        # Train the models
        predictor.train(model_names)

        # Get scores
        scores_df = predictor.get_scores()
        scores = scores_df.to_dict('records')

        return jsonify({
            'success': True,
            'message': 'Training completed',
            'scores': scores,
            'trained_models': list(predictor.trained_models.keys())
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using a trained model"""
    global predictor

    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'No predictor initialized. Please load data first.'
        }), 400

    try:
        data = request.get_json()
        model_name = data.get('model', 'RandomForest')
        input_data = data.get('data')

        if model_name not in predictor.trained_models:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not trained. Please train it first.'
            }), 400

        # Convert input data to DataFrame
        if isinstance(input_data, dict):
            # Single prediction
            X_new = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            # Multiple predictions
            X_new = pd.DataFrame(input_data)
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid input data format'
            }), 400

        # Ensure columns match training data
        X_new = X_new[predictor.X.columns]

        # Make predictions
        predictions = predictor.predict(model_name, X_new)

        return jsonify({
            'success': True,
            'model': model_name,
            'predictions': predictions.tolist(),
            'count': len(predictions)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/scores')
def get_scores():
    """Get RMSE scores for all trained models"""
    global predictor

    if predictor is None or not predictor.trained_models:
        return jsonify({
            'success': False,
            'error': 'No trained models available'
        }), 400

    scores_df = predictor.get_scores()
    return jsonify({
        'success': True,
        'scores': scores_df.to_dict('records')
    })

@app.route('/api/save-model', methods=['POST'])
def save_trained_model():
    """Save a trained model to disk"""
    global predictor

    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'No predictor initialized'
        }), 400

    try:
        data = request.get_json()
        model_name = data.get('model')

        if model_name not in predictor.trained_models:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} not trained'
            }), 400

        filename = save_model(model_name, predictor)

        return jsonify({
            'success': True,
            'message': f'Model saved successfully',
            'filename': filename
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/list-saved-models')
def list_saved_models():
    """List all saved models"""
    try:
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        files.sort(reverse=True)  # Most recent first

        return jsonify({
            'success': True,
            'models': files,
            'count': len(files)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/dataset-info')
def dataset_info():
    """Get information about the loaded dataset"""
    global df, predictor

    if df is None:
        return jsonify({
            'success': False,
            'error': 'No data loaded'
        }), 400

    try:
        info = {
            'success': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'statistics': df.describe().to_dict()
        }

        if predictor:
            info['target'] = predictor.target
            info['train_size'] = predictor.X_train.shape
            info['test_size'] = predictor.X_test.shape

        return jsonify(info)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ==================== Run App ====================

if __name__ == '__main__':
    print("=" * 50)
    print("Apartment Hunter - Price Predictor API")
    print("=" * 50)
    print("\nAvailable endpoints:")
    print("  GET  /                    - Home page")
    print("  GET  /api/status          - Check application status")
    print("  POST /api/load-data       - Load dataset")
    print("  POST /api/train           - Train models")
    print("  POST /api/predict         - Make predictions")
    print("  GET  /api/scores          - Get model scores")
    print("  POST /api/save-model      - Save trained model")
    print("  GET  /api/list-saved-models - List saved models")
    print("  GET  /api/dataset-info    - Get dataset information")
    print("\n" + "=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)
