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
