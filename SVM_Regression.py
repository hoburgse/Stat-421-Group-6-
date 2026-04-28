# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:14:02 2026

@author: ajenn
"""

# Import Libraries that are needed
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import Dataset 
# Load data
df = pd.read_csv("C:/Users/ajenn/Downloads/student_stress_sleep_screen.csv")

# Shape the dataset variables
X = df[['sleep_hours', 'screen_time_hours', 'caffeine_intake', 'age']]
y = df['study_hours']

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=55)

# SVR model with scaling
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])

# Evaluate with R^2
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("R^2:", scores.mean())







