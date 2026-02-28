"""
Loan Default Prediction - Random Forest Model
A complete pipeline for training and evaluating a loan default prediction model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print('=' * 60)
print('LOAN DEFAULT PREDICTION - MODEL TRAINING')
print('=' * 60)

# ============== 1. LOAD DATASET ==============
print('\n[Step 1] Loading Dataset...')
csv_path = os.path.join(os.getcwd(), 'Loan_default.csv')
print(f'Loading: {csv_path}')
df = pd.read_csv(csv_path)
print(f'Dataset Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'\nFirst few rows:')
print(df.head())

# ============== 2. PREPROCESSING ==============
print('\n[Step 2] Preprocessing...')
print('Missing values per column:')
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()
print(f'Shape after dropping NaN: {df.shape}')

# Clean column names
df.columns = df.columns.str.strip()

# Drop ID columns
id_cols = [c for c in df.columns if 'id' in c.lower()]
if id_cols:
    print(f'Dropping ID columns: {id_cols}')
    df = df.drop(columns=id_cols)

# Identify target column
possible_targets = ['Loan_Status', 'default', 'Loan_Default', 'Target', 'loan_status', 'LoanStatus', 'loan_default', 'Default']
target = None
for t in possible_targets:
    if t in df.columns:
        target = t
        break

if target is None:
    for c in df.columns:
        if df[c].dtype == 'object' and df[c].nunique() == 2:
            target = c
            break

if target is None:
    raise ValueError('Cannot find target column automatically.')

print(f'Using target column: {target}')

# Convert target to numeric (0/1)
df[target] = pd.factorize(df[target])[0]

# Define features and target
X = df.drop(columns=[target])
y = df[target]

print(f'Feature matrix shape: {X.shape}')
print(f'Target distribution:\n{y.value_counts()}')

# ============== 3. TRAIN-TEST SPLIT ==============
print('\n[Step 3] Train-Test Split...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)
print(f'Training set: {X_train.shape}')
print(f'Test set: {X_test.shape}')

# ============== 4. BUILD PIPELINE ==============
print('\n[Step 4] Building ML Pipeline...')

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f'Numeric columns: {numeric_cols}')
print(f'Categorical columns: {categorical_cols}')

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

# ============== 5. TRAIN MODEL ==============
print('\n[Step 5] Training Model...')
pipeline.fit(X_train, y_train)
print('✅ Model training completed')

# ============== 6. EVALUATE MODEL ==============
print('\n[Step 6] Model Evaluation...')
y_pred = pipeline.predict(X_test)

print('\n=== RANDOM FOREST METRICS ===')
print(f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}')
print(f'Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}')
print(f'F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}')
print(f'\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# ============== 7. COMPARE WITH OTHER MODELS ==============
print('\n[Step 7] Comparing with Other Models...')
print('\n=== Cross-Validation Scores (5-Fold) ===')

base_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

cv_scores = {}
for name, model in base_models.items():
    pipeline_temp = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    scores = cross_val_score(pipeline_temp, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores[name] = scores
    print(f'{name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})')

# Train Gradient Boosting as it often performs better
print('\n[Step 8] Training Improved Model (Gradient Boosting)...')
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

best_pipeline.fit(X_train, y_train)
y_pred_best = best_pipeline.predict(X_test)

print('\n=== GRADIENT BOOSTING METRICS ===')
print(f'Accuracy:  {accuracy_score(y_test, y_pred_best):.4f}')
print(f'Precision: {precision_score(y_test, y_pred_best, zero_division=0):.4f}')
print(f'Recall:    {recall_score(y_test, y_pred_best, zero_division=0):.4f}')
print(f'F1 Score:  {f1_score(y_test, y_pred_best, zero_division=0):.4f}')
print(f'\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred_best))

# ============== 9. SAVE MODELS ==============
print('\n[Step 9] Saving Models...')
joblib.dump(pipeline, 'random_forest_pipeline.pkl')
print('✅ Random Forest model saved: random_forest_pipeline.pkl')

joblib.dump(best_pipeline, 'gradient_boosting_pipeline.pkl')
print('✅ Gradient Boosting model saved: gradient_boosting_pipeline.pkl')

print('\n' + '=' * 60)
print('MODEL TRAINING COMPLETE')
print('=' * 60)
