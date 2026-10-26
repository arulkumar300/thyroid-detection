import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")
df_raw = pd.read_csv(r'C:\Users\elango\Downloads\thyroidDF (4).csv')
print("Shape:", df_raw.shape)

# Target mapping
full_name_map = {0: 'Negative', 1: 'Hypothyroid', 2: 'Hyperthyroid'}
target_col = 'target'
raw_vals = df_raw[target_col].unique()
label_map = {}
for val in raw_vals:
    v = str(val).strip().lower().replace(' ','_')
    v_orig = str(val).strip()
    if any(x in v for x in ['neg','normal','euthyroid','none','-','false']):
        label_map[val] = 0
    elif any(x in v for x in ['hypo','compensated','primary_h','secondary_h','subnormal']):
        label_map[val] = 1
    elif any(x in v for x in ['hyper','toxic','goitre','graves','t3_toxic']):
        label_map[val] = 2
    elif v_orig in ['A','B','C','D','K','AK','BK','CK','DK','GK','FK','MK','KJ']:
        label_map[val] = 1
    elif v_orig in ['E','F','G','H','I','J','L','M','N','O','P','Q','R','S','GI','OI','LJ','GKJ']:
        label_map[val] = 2
    elif v_orig == '-':
        label_map[val] = 0
    else:
        try:
            label_map[val] = int(float(v_orig)) % 3
        except:
            label_map[val] = 0

df_raw['label'] = df_raw[target_col].map(label_map)
print("Class counts:\n", df_raw['label'].value_counts())

FEATURE_NAMES = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
    'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured'
]

# Map categorical columns to float/numeric
for col in df_raw.columns:
    if col == 'label': continue
    if df_raw[col].dtype == 'object':
        df_raw[col] = df_raw[col].map(
            {'t':1,'f':0,'T':1,'F':0,'M':1,'m':1,
             'y':1,'n':0,'Y':1,'N':0}
        ).fillna(df_raw[col])
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

X = df_raw[FEATURE_NAMES]
y = df_raw['label'].values

print("X shape:", X.shape)
print("y unique values and counts:", np.unique(y, return_counts=True))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Imputation
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc = scaler.transform(X_test_imp)

# Oversampling minority classes using pure pandas/numpy (alternative to SMOTE since imblearn is not installed)
# Identify indices
train_df = pd.DataFrame(X_train_sc, columns=FEATURE_NAMES)
train_df['label'] = y_train

max_size = train_df['label'].value_counts().max()
lst = [train_df]
for class_index, group in train_df.groupby('label'):
    if len(group) < max_size:
        # oversample with replacement
        oversampled = group.sample(max_size - len(group), replace=True, random_state=42)
        lst.append(oversampled)

train_df_balanced = pd.concat(lst).sample(frac=1.0, random_state=42)
X_train_bal = train_df_balanced.drop(columns=['label']).values
y_train_bal = train_df_balanced['label'].values

print("Balanced X shape:", X_train_bal.shape)
print("Balanced class distribution:", np.bincount(y_train_bal))

# Feature selection
selector = SelectKBest(mutual_info_classif, k=15)
X_train_sel = selector.fit_transform(X_train_bal, y_train_bal)
X_test_sel = selector.transform(X_test_sc)

selected_features = [FEATURE_NAMES[i] for i in selector.get_support(indices=True)]
print("Selected features:", selected_features)

# Train Stacking model
print("Training stacking model...")
stacking_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    stack_method='predict_proba'
)
stacking_model.fit(X_train_sel, y_train_bal)

y_pred = stacking_model.predict(X_test_sel)
print("Stacking Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Hypothyroid', 'Hyperthyroid']))
