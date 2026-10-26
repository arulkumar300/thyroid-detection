from app import app
import json

client = app.test_client()

def test(name, payload):
    resp = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    result = resp.get_json()
    print(f"  [{name}] => {result['diagnosis']} (confidence: {result['confidence']})")

print("=== TEST 1: Euthyroid (Normal patient) ===")
test("Normal", {
    "age": 45, "sex": "F",
    "on_thyroxine": "No", "query_on_thyroxine": "No",
    "on_antithyroid_medication": "No", "sick": "No", "pregnant": "No",
    "thyroid_surgery": "No", "I131_treatment": "No",
    "query_hypothyroid": "No", "query_hyperthyroid": "No",
    "lithium": "No", "goitre": "No", "tumor": "No",
    "hypopituitary": "No", "psych": "No",
    "TSH": 2.5, "T3": 1.5, "TT4": 100, "T4U": 1.0, "FTI": 110
})

print("\n=== TEST 2: Hypothyroid (High TSH, Low T4) ===")
test("Hypothyroid", {
    "age": 55, "sex": "F",
    "on_thyroxine": "No", "query_on_thyroxine": "No",
    "on_antithyroid_medication": "No", "sick": "Yes", "pregnant": "No",
    "thyroid_surgery": "No", "I131_treatment": "No",
    "query_hypothyroid": "Yes", "query_hyperthyroid": "No",
    "lithium": "No", "goitre": "Yes", "tumor": "No",
    "hypopituitary": "No", "psych": "No",
    "TSH": 15.0, "T3": 0.5, "TT4": 40, "T4U": 0.8, "FTI": 45
})

print("\n=== TEST 3: Hyperthyroid (Low TSH, High T3/T4) ===")
test("Hyperthyroid", {
    "age": 38, "sex": "F",
    "on_thyroxine": "No", "query_on_thyroxine": "No",
    "on_antithyroid_medication": "No", "sick": "No", "pregnant": "No",
    "thyroid_surgery": "No", "I131_treatment": "No",
    "query_hypothyroid": "No", "query_hyperthyroid": "Yes",
    "lithium": "No", "goitre": "Yes", "tumor": "No",
    "hypopituitary": "No", "psych": "No",
    "TSH": 0.1, "T3": 3.5, "TT4": 200, "T4U": 1.3, "FTI": 185
})

print("\n=== ALL TESTS COMPLETE ===")
