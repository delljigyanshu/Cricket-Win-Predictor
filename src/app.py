# src/app.py
import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Flask: templates in ../templates, static in ../static
app = Flask(__name__, template_folder='../templates', static_folder='../static')

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
LR_PATH = os.path.join(MODEL_DIR, 'logistic.joblib')
GB_CAL_PATH = os.path.join(MODEL_DIR, 'gb_calibrated.joblib')
GB_PATH = os.path.join(MODEL_DIR, 'gb.joblib')

# Try to load models (may be missing if not trained)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
lr = joblib.load(LR_PATH) if os.path.exists(LR_PATH) else None
if os.path.exists(GB_CAL_PATH):
    gb = joblib.load(GB_CAL_PATH)
elif os.path.exists(GB_PATH):
    gb = joblib.load(GB_PATH)
else:
    gb = None

# Feature order expected by model & frontend
FEATURE_ORDER = [
    'score_before','wickets_before','overs_completed','balls_remaining','runs_required',
    'req_run_rate','current_run_rate','wickets_in_hand','frac_innings_complete','runs_required_norm',
    'rr_diff','pressure','over_int','batsman_form','bowler_form'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error":"No JSON body provided"}), 400

    # Build numeric feature vector (in FEATURE_ORDER)
    try:
        x = np.array([float(data.get(k, 0.0)) for k in FEATURE_ORDER]).reshape(1, -1)
    except Exception as e:
        return jsonify({"error":"Bad input data", "detail": str(e)}), 400

    # Logistic uses scaled features
    p_lr = None
    p_gb = None

    if scaler is not None and lr is not None:
        try:
            x_s = scaler.transform(x)
            p_lr = float(lr.predict_proba(x_s)[0,1])
        except Exception as e:
            p_lr = None

    if gb is not None:
        try:
            # gb may be a calibrated classifier or raw model
            p_gb = float(gb.predict_proba(x)[0,1]) if hasattr(gb, 'predict_proba') else float(gb.predict_proba(x_s)[0,1])
        except Exception:
            try:
                p_gb = float(gb.predict_proba(x_s)[0,1])
            except Exception:
                p_gb = None

    if p_lr is None and p_gb is None:
        return jsonify({"error":"No models available on server. Train models and save to models/"}), 500

    # Combine probabilities (simple average when both exist)
    if p_lr is not None and p_gb is not None:
        combined = (p_lr + p_gb) / 2.0
    else:
        combined = p_gb if p_gb is not None else p_lr

    # Return percentages rounded to 2 decimals
    result = {}
    if p_lr is not None:
        result['lr'] = round(p_lr * 100.0, 2)
    if p_gb is not None:
        result['gb'] = round(p_gb * 100.0, 2)
    result['combined'] = round(combined * 100.0, 2)

    return jsonify(result)

if __name__ == '__main__':
    # set host 0.0.0.0 to be reachable on local network if needed
    app.run(debug=True, host='0.0.0.0', port=5000)
