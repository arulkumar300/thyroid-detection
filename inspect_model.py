import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

m = joblib.load('thyroid_model__3_.pkl')

print("=== MODEL TYPE ===")
print(type(m))
print()

print("=== ALL ATTRIBUTES ===")
for attr in dir(m):
    if not attr.startswith('__'):
        try:
            val = getattr(m, attr)
            if not callable(val):
                print(f"  {attr}: {type(val).__name__} = {repr(val)[:200]}")
        except:
            pass

print()
print("=== ESTIMATORS ===")
if hasattr(m, 'estimators_'):
    for i, est in enumerate(m.estimators_):
        if isinstance(est, tuple):
            print(f"  [{i}] name={est[0]}, type={type(est[1])}")
        else:
            print(f"  [{i}] type={type(est)}")

print()
print("=== FINAL ESTIMATOR ===")
if hasattr(m, 'final_estimator_'):
    print(f"  type={type(m.final_estimator_)}")
    print(f"  {m.final_estimator_}")

print()
print("=== CLASSES ===")
if hasattr(m, 'classes_'):
    print(f"  classes_: {m.classes_}")

print()
print("=== LABEL ENCODER CHECK ===")
# Check if there's a label encoder stored
for attr in dir(m):
    val = getattr(m, attr, None)
    if val is not None and 'LabelEncoder' in str(type(val)):
        print(f"  Found LabelEncoder at m.{attr}")
        print(f"  classes_: {val.classes_}")

print()
print("=== N_FEATURES_IN_ ===")
print(f"  {getattr(m, 'n_features_in_', 'NOT FOUND')}")

print()
print("=== TESTING PREDICTIONS ===")
# Test with various extreme inputs to see what classes come out
test_cases = [
    ("All zeros", np.zeros((1, 15))),
    ("High feature[3]", np.array([[0,0,0,10,0,0,0,0,0,0,0,0,0,0,0]])),
    ("High feature[8]", np.array([[0,0,0,0,0,0,0,0,15,0,0,0,0,0,0]])),
    ("Known class 1 input", np.array([[ 0.80300065, -0.31063969, -0.46777139, -2.19910478, -1.32076782, -0.37969381, 0.37101494, 1.01011528, 0.10170996, 1.74614579, 1.44841959, -0.76516476, 0.08280418, -0.22526464, 0.36410931]])),
]

for name, x in test_cases:
    pred = m.predict(x)
    proba = m.predict_proba(x) if hasattr(m, 'predict_proba') else None
    print(f"  {name}: pred={pred[0]}, proba={proba[0] if proba is not None else 'N/A'}")

print()
print("=== CHECKING IF PKL IS A DICT/TUPLE ===")
# Sometimes pkl files contain multiple objects
m2 = joblib.load('thyroid_model__3_.pkl')
if isinstance(m2, dict):
    print("  It's a dict! Keys:", list(m2.keys()))
elif isinstance(m2, (list, tuple)):
    print("  It's a list/tuple! Length:", len(m2))
    for i, item in enumerate(m2):
        print(f"    [{i}] {type(item)}")
else:
    print(f"  It's a single object: {type(m2)}")
