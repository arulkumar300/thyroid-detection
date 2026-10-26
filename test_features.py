import joblib

imputer = joblib.load('thyroid_imputer.pkl')
print(list(imputer.feature_names_in_))
