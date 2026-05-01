# -*- coding: utf-8 -*-
"""
Created on Fri May 1 2026

@author: martinezjg
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('student_stress_sleep_screen.csv')

# Encode categoricals
for col in ['gender', 'physical_activity', 'academic_pressure', 'stress_level']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df[['age', 'gender', 'sleep_hours', 'screen_time_hours',
         'study_hours', 'physical_activity', 'caffeine_intake', 'academic_pressure']]
y = df['stress_level']

# 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
print("CV Accuracy:", cross_val_score(dt, X, y, cv=kf).mean())

# Confusion matrix per fold
for i, (train, test) in enumerate(kf.split(X)):
    dt.fit(X.iloc[train], y.iloc[train])
    preds = dt.predict(X.iloc[test])
    print(f"\nFold {i+1} Confusion Matrix:")
    print(confusion_matrix(y.iloc[test], preds))

# Plot tree
dt.fit(X, y)
plt.figure(figsize=(18, 6))
plot_tree(dt, feature_names=X.columns, filled=True, rounded=True, fontsize=9)
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.show() 

