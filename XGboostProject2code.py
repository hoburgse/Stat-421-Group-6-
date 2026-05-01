# -*- coding: utf-8 -*-
"""
Created on Fri May 1 08:06:51 2026

@author: naolm

This program uses XGBoost models to:
1. Predict sleep hours (regression)
2. Predict stress level (classification)
"""

# Import libraries needed for data handling and models
import pandas as pd
import numpy as np

# Import XGBoost models
from xgboost import XGBRegressor, XGBClassifier

# Import tools for model testing
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# Import tool to convert text categories into numbers
from sklearn.preprocessing import LabelEncoder

# Import plotting library (may be used later)
import matplotlib.pyplot as plt


# -------------------------
# Load dataset
# -------------------------

# Read the CSV file into a dataframe
df = pd.read_csv(
    r"C:\Users\naolm\Downloads\student_stress_sleep_screen.csv"
)


# -------------------------
# Convert text columns into numbers
# -------------------------

# These columns contain text values
categorical_cols = [
    'gender',
    'physical_activity',
    'academic_pressure'
]

# Convert each text column into numeric values
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


# -------------------------
# Remove student ID column
# -------------------------

# student_id is not useful for prediction
df = df.drop(columns=['student_id'])


# ====================================================
# Model 1: Predict Sleep Hours (Regression)
# ====================================================

# Features (input data)
# Remove sleep_hours and stress_level from inputs
X_sleep = df.drop(
    columns=['sleep_hours', 'stress_level']
)

# Target (what we want to predict)
y_sleep = df['sleep_hours']


# Create 5-fold cross validation
# This splits data into 5 groups for testing
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Create XGBoost regression model
sleep_model = XGBRegressor(
    n_estimators=100,   # number of trees
    max_depth=4,        # tree size
    learning_rate=0.1,  # learning speed
    random_state=42
)


# Test the model using cross validation
sleep_scores = cross_val_score(
    sleep_model,
    X_sleep,
    y_sleep,
    cv=kf,
    scoring='r2'   # use R² score
)


# Print average R² score
print("XGBoost Sleep Prediction")
print("Predictive R²:",
      round(sleep_scores.mean(), 4))


# ====================================================
# Model 2: Predict Stress Level (Classification)
# ====================================================

# Features (input data)
X_stress = df.drop(columns=['stress_level'])

# Target column
y_stress = df['stress_level']


# Convert stress levels into numbers
stress_encoder = LabelEncoder()
y_stress_encoded = stress_encoder.fit_transform(y_stress)


# Use stratified split to keep class balance
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Create XGBoost classification model
stress_model = XGBClassifier(
    n_estimators=100,   # number of trees
    max_depth=4,        # tree size
    learning_rate=0.1,  # learning speed
    random_state=42,
    eval_metric='mlogloss'
)


# Test classifier using cross validation
stress_scores = cross_val_score(
    stress_model,
    X_stress,
    y_stress_encoded,
    cv=skf,
    scoring='accuracy'
)


# Print average accuracy score
print("\nXGBoost Stress Prediction")
print("Accuracy:",
      round(stress_scores.mean(), 4))

# Show what each stress number means
print("Stress Classes:",
      list(stress_encoder.classes_))