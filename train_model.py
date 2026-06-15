<<<<<<< HEAD
import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'data', 'huge_colleges.csv')

df = pd.read_csv(csv_path)

# Supported schemas:
# 1) Old synthetic schema (this file used earlier):
#    ['college_name','cutoff_percentage','caste','branch','gender','quota']
# 2) New dataset schema:
#    ['college_name','choice_code','branch','category_code','percentile','stage','year']

if {'cutoff_percentage', 'caste', 'gender', 'quota'}.issubset(df.columns):
    # --- Old schema ---
    le_caste = LabelEncoder()
    df['caste_code'] = le_caste.fit_transform(df['caste'].astype(str))

    le_branch = LabelEncoder()
    df['branch_code'] = le_branch.fit_transform(df['branch'].astype(str))

    le_gender = LabelEncoder()
    df['gender_code'] = le_gender.fit_transform(df['gender'].astype(str))

    le_quota = LabelEncoder()
    df['quota_code'] = le_quota.fit_transform(df['quota'].astype(str))

    X = df[['cutoff_percentage', 'caste_code', 'branch_code', 'gender_code', 'quota_code']].values
    y = df['college_name'].astype(str).values

else:
    # --- New schema ---
    required = {'college_name', 'branch', 'category_code', 'percentile'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"huge_colleges.csv missing required columns for new schema: {sorted(missing)}")

    # Map dataset's category_code into the existing app's expected 'caste' input
    # (so we can keep app.py and UI unchanged).
    # category_code values look like: GOPEN, GSC, GST, GOBC, ...
    df['caste'] = df['category_code'].astype(str)
    df['cutoff_percentage'] = df['percentile'].astype(float)

    # App currently expects gender/quota too, but dataset doesn't have them.
    # Use constant placeholders so encoders/model accept the columns.
    df['gender'] = 'M'
    df['quota'] = 'MS'

    le_caste = LabelEncoder()
    df['caste_code'] = le_caste.fit_transform(df['caste'])

    le_branch = LabelEncoder()
    df['branch_code'] = le_branch.fit_transform(df['branch'].astype(str))

    le_gender = LabelEncoder()
    df['gender_code'] = le_gender.fit_transform(df['gender'].astype(str))

    le_quota = LabelEncoder()
    df['quota_code'] = le_quota.fit_transform(df['quota'].astype(str))

    X = df[['cutoff_percentage', 'caste_code', 'branch_code', 'gender_code', 'quota_code']].values
    y = df['college_name'].astype(str).values


le_college = LabelEncoder()
y_encoded = le_college.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# HistGradientBoosting tends to yield smoother, better-calibrated multiclass
# probabilities than RandomForest on this tabular setup (few features, many classes).
model = HistGradientBoostingClassifier(
    max_iter=500,
    max_depth=14,
    learning_rate=0.06,
    l2_regularization=0.05,
    min_samples_leaf=15,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=25,
)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_proba)

dump(model, os.path.join(BASE_DIR, 'model_huge.joblib'))
dump(le_college, os.path.join(BASE_DIR, 'college_encoder_huge.joblib'))
dump(le_caste, os.path.join(BASE_DIR, 'caste_encoder_huge.joblib'))
dump(le_branch, os.path.join(BASE_DIR, 'branch_encoder_huge.joblib'))
dump(le_gender, os.path.join(BASE_DIR, 'gender_encoder_huge.joblib'))
dump(le_quota, os.path.join(BASE_DIR, 'quota_encoder_huge.joblib'))

print('Model: HistGradientBoostingClassifier')
print('Huge model trained with caste + branch + gender + quota.')
print(f'Num castes: {len(le_caste.classes_)}')
print(f'Num branches: {len(le_branch.classes_)}')
print(f'Num colleges: {len(le_college.classes_)}')
print(f'Num genders: {len(le_gender.classes_)}')
print(f'Num quotas: {len(le_quota.classes_)}')
print(f'Hold-out accuracy: {acc:.4f}')
print(f'Hold-out log loss (lower is better): {ll:.4f}')

df.to_pickle(os.path.join(BASE_DIR, 'huge_colleges_data.pkl'))
=======
import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'data', 'huge_colleges.csv')

df = pd.read_csv(csv_path)

le_caste = LabelEncoder()
df['caste_code'] = le_caste.fit_transform(df['caste'])

le_branch = LabelEncoder()
df['branch_code'] = le_branch.fit_transform(df['branch'])

le_gender = LabelEncoder()
df['gender_code'] = le_gender.fit_transform(df['gender'])

le_quota = LabelEncoder()
df['quota_code'] = le_quota.fit_transform(df['quota'])

X = df[['cutoff_percentage', 'caste_code', 'branch_code', 'gender_code', 'quota_code']].values
y = df['college_name'].values

le_college = LabelEncoder()
y_encoded = le_college.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Candidate models for better realism/generalization.
candidates = [
    (
        "balanced_depth",
        HistGradientBoostingClassifier(
            max_iter=650,
            max_depth=12,
            learning_rate=0.05,
            l2_regularization=0.08,
            min_samples_leaf=25,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=30,
        ),
    ),
    (
        "wider_trees",
        HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=14,
            learning_rate=0.06,
            l2_regularization=0.05,
            min_samples_leaf=15,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=25,
        ),
    ),
]

best = None
best_score = -1e9
model = None
metrics = {}

for name, candidate in candidates:
    candidate.fit(X_train, y_train)
    y_proba = candidate.predict_proba(X_test)
    y_pred = candidate.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    top3 = top_k_accuracy_score(y_test, y_proba, k=3, labels=np.arange(len(le_college.classes_)))
    top5 = top_k_accuracy_score(y_test, y_proba, k=5, labels=np.arange(len(le_college.classes_)))
    # Higher is better; emphasize shortlist quality (top-3/top-5) and calibration (logloss).
    score = (0.45 * top3) + (0.35 * top5) + (0.20 * acc) - (0.05 * ll)
    metrics[name] = {
        "accuracy": float(acc),
        "log_loss": float(ll),
        "top3_accuracy": float(top3),
        "top5_accuracy": float(top5),
        "score": float(score),
    }
    if score > best_score:
        best_score = score
        best = name
        model = candidate

dump(model, os.path.join(BASE_DIR, 'model_huge.joblib'))
dump(le_college, os.path.join(BASE_DIR, 'college_encoder_huge.joblib'))
dump(le_caste, os.path.join(BASE_DIR, 'caste_encoder_huge.joblib'))
dump(le_branch, os.path.join(BASE_DIR, 'branch_encoder_huge.joblib'))
dump(le_gender, os.path.join(BASE_DIR, 'gender_encoder_huge.joblib'))
dump(le_quota, os.path.join(BASE_DIR, 'quota_encoder_huge.joblib'))

print('Model: HistGradientBoostingClassifier')
print('Huge model trained with caste + branch + gender + quota.')
print(f'Num castes: {len(le_caste.classes_)}')
print(f'Num branches: {len(le_branch.classes_)}')
print(f'Num colleges: {len(le_college.classes_)}')
print(f'Num genders: {len(le_gender.classes_)}')
print(f'Num quotas: {len(le_quota.classes_)}')
print(f'Selected variant: {best}')
for name in metrics:
    m = metrics[name]
    print(
        f"[{name}] acc={m['accuracy']:.4f} top3={m['top3_accuracy']:.4f} "
        f"top5={m['top5_accuracy']:.4f} logloss={m['log_loss']:.4f} score={m['score']:.4f}"
    )

df.to_pickle(os.path.join(BASE_DIR, 'huge_colleges_data.pkl'))
>>>>>>> 9d8d72e274a1cb006ab020cb5aeb4faabc27fd48
