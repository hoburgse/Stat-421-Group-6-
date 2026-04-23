#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:33:52 2026

@author: hoburgse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
# Ensure StessSleep.csv is in the same directory as this script
df = pd.read_csv('StessSleep.csv')

# 2. Preprocessing
# Drop non-predictive columns
df_model = df.drop(columns=['student_id'])

# Map 'stress_level' to numeric values for Regression
# Low = 0, Medium = 1, High = 2
stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df_model['stress_level'] = df_model['stress_level'].map(stress_mapping)

# Encode other categorical columns (Gender, Physical Activity, Academic Pressure)
le = LabelEncoder()
for col in ['gender', 'physical_activity', 'academic_pressure']:
    df_model[col] = le.fit_transform(df_model[col])

# Define Features (X) and Target (y)
X = df_model.drop(columns=['stress_level'])
y = df_model['stress_level']

# 3. Initialize Random Forest Regressor and Cross-Validation
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Perform 5-Fold Cross Validation
# 'cross_val_predict' returns the predicted value for each point when it was in the test set
y_pred_continuous = cross_val_predict(rf_regressor, X, y, cv=kf)

# 5. Calculate Regression Metric: R-squared
r2 = r2_score(y, y_pred_continuous)

# 6. Generate Confusion Matrix Metrics
# To create a confusion matrix, we round the continuous predictions to the nearest integer
y_pred_discrete = np.clip(np.round(y_pred_continuous), 0, 2).astype(int)
cm = confusion_matrix(y, y_pred_discrete)

# 7. Print Results
print(f"--- Model Evaluation ---")
print(f"R-squared Score (Regression): {r2:.4f}")
print(f"Accuracy (after rounding): {accuracy_score(y, y_pred_discrete):.4f}")
print("\nConfusion Matrix:")
print(cm)

# 8. Visualization (Plots will appear in the Spyder 'Plots' pane)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'], 
            yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Random Forest Regression: Confusion Matrix\n$R^2$ Score: {r2:.3f}')
plt.show()