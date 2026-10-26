from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import traceback
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Load pipeline components
imputer = None
scaler = None
selector = None
model = None

try:
    imputer = joblib.load('thyroid_imputer.pkl')
    scaler = joblib.load('thyroid_scaler.pkl')
    selector = joblib.load('thyroid_selector.pkl')
    model = joblib.load('thyroid_model.pkl')
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")

FEATURE_NAMES = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
    'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured'
]

CLASS_MAP = {0: 'Euthyroid', 1: 'Hypothyroid', 2: 'Hyperthyroid'}

def parse_bool(val):
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ['yes', 'true', 'male', 'm', '1', 't', 'y']:
            return 1.0
        elif v in ['no', 'false', 'female', 'f', '0', 'n']:
            return 0.0
    try:
        f = float(val)
        return f if f in [0.0, 1.0] else np.nan
    except:
        return np.nan

def parse_float(val):
    if val is None or val == '':
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model pipeline not loaded'}), 500

    try:
        data = request.json

        # Parse categorical / boolean fields
        row = {}
        row['age'] = parse_float(data.get('age'))
        row['sex'] = parse_bool(data.get('sex'))
        row['on_thyroxine'] = parse_bool(data.get('on_thyroxine'))
        row['query_on_thyroxine'] = parse_bool(data.get('query_on_thyroxine'))
        
        # NOTE: the frontend might send 'on_antithyroid_medication' instead of 'on_antithyroid_meds'
        row['on_antithyroid_meds'] = parse_bool(data.get('on_antithyroid_medication', data.get('on_antithyroid_meds')))
        row['sick'] = parse_bool(data.get('sick'))
        row['pregnant'] = parse_bool(data.get('pregnant'))
        row['thyroid_surgery'] = parse_bool(data.get('thyroid_surgery'))
        row['I131_treatment'] = parse_bool(data.get('I131_treatment'))
        row['query_hypothyroid'] = parse_bool(data.get('query_hypothyroid'))
        row['query_hyperthyroid'] = parse_bool(data.get('query_hyperthyroid'))
        row['lithium'] = parse_bool(data.get('lithium'))
        row['goitre'] = parse_bool(data.get('goitre'))
        row['tumor'] = parse_bool(data.get('tumor'))
        row['hypopituitary'] = parse_bool(data.get('hypopituitary'))
        row['psych'] = parse_bool(data.get('psych'))

        # Continuous fields and their measured flags
        for f in ['TSH', 'T3', 'TT4', 'T4U', 'FTI']:
            val = parse_float(data.get(f))
            row[f] = val
            row[f'{f}_measured'] = 0.0 if np.isnan(val) else 1.0
            
        row['TBG_measured'] = 0.0 # TBG is always missing based on our feature set

        # Build DataFrame in exact feature order
        df_raw = pd.DataFrame([row], columns=FEATURE_NAMES)
        
        # Pipeline steps
        # Manual imputation to avoid sklearn version mismatch issue with SimpleImputer
        imputer_stats = [54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.4, 1.0, 2.0, 1.0, 104.0, 1.0, 0.98, 1.0, 107.0, 0.0]
        imputer_dict = dict(zip(FEATURE_NAMES, imputer_stats))
        df_imp = df_raw.fillna(imputer_dict)
        
        arr_sc = scaler.transform(df_imp)
        arr_sel = selector.transform(arr_sc)

        prediction = model.predict(arr_sel)[0]
        proba = model.predict_proba(arr_sel)[0]
        
        final_diagnosis = CLASS_MAP.get(prediction, "Unknown")
        confidence = float(np.max(proba))

        print(f"Prediction: {prediction} ({final_diagnosis}), Confidence: {confidence:.4f}")

        return jsonify({
            'diagnosis': final_diagnosis,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(port=5000, debug=True)
