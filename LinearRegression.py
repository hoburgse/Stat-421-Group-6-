# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:47:52 2026

@author: Abhinav
"""

"""
Linear Regression Analysis — Student Stress, Sleep, and Screen Time
Dataset: 500 students, 10 variables
Target: stress_level (encoded as Low=0, Medium=1, High=2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── 1. Load data ──────────────────────────────────────────────────────────────

df = pd.read_csv("student_stress_sleep_screen.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ── 2. Encode categorical variables ──────────────────────────────────────────

# Ordinal encode target
stress_map = {"Low": 0, "Medium": 1, "High": 2}
df["stress_num"] = df["stress_level"].map(stress_map)

# Binary / nominal encode predictors
df["gender_enc"]            = LabelEncoder().fit_transform(df["gender"])
df["physical_activity_enc"] = LabelEncoder().fit_transform(df["physical_activity"])
df["academic_pressure_enc"] = LabelEncoder().fit_transform(df["academic_pressure"])

# ── 3. Define features and target ─────────────────────────────────────────────

FEATURES = [
    "age",
    "sleep_hours",
    "screen_time_hours",
    "study_hours",
    "caffeine_intake",
    "gender_enc",
    "physical_activity_enc",
    "academic_pressure_enc",
]

X = df[FEATURES]
y = df["stress_num"]

print("\nFeature matrix shape:", X.shape)
print("Target distribution:\n", y.value_counts().sort_index())

# ── 4. Standardise features ───────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 5. Train / test split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 6. Fit linear regression ──────────────────────────────────────────────────

model = LinearRegression()
model.fit(X_train, y_train)

# ── 7. Predict and round to nearest class ─────────────────────────────────────

y_pred_cont  = model.predict(X_test)
y_pred_class = np.round(y_pred_cont).clip(0, 2).astype(int)

# ── 8. Regression metrics ─────────────────────────────────────────────────────

mse  = mean_squared_error(y_test, y_pred_cont)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred_cont)
r2   = r2_score(y_test, y_pred_cont)
acc  = (y_pred_class == y_test.values).mean()

print("\n── Regression metrics (test set) ──")
print(f"  R²   : {r2:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  Classification accuracy (rounded predictions): {acc:.4f}")

# ── 9. 5-fold cross-validation ────────────────────────────────────────────────

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2  = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
cv_mse = cross_val_score(model, X_scaled, y, cv=kf, scoring="neg_mean_squared_error")

print("\n── 5-Fold cross-validation ──")
for i, (r2_fold, mse_fold) in enumerate(zip(cv_r2, -cv_mse), 1):
    print(f"  Fold {i}: R² = {r2_fold:.4f}  RMSE = {np.sqrt(mse_fold):.4f}")
print(f"\n  Mean R²   : {cv_r2.mean():.4f}  (±{cv_r2.std():.4f})")
print(f"  Mean RMSE : {np.sqrt(-cv_mse.mean()):.4f}  (±{np.sqrt(-cv_mse).std():.4f})")

# ── 10. Coefficients ──────────────────────────────────────────────────────────

coef_df = pd.DataFrame(
    {"Feature": FEATURES, "Coefficient": model.coef_}
).sort_values("Coefficient", key=abs, ascending=False)

print(f"\n── Model coefficients (intercept = {model.intercept_:.4f}) ──")
print(coef_df.to_string(index=False))

# ── 11. Confusion matrix (rounded predictions) ────────────────────────────────

label_names = ["Low", "Medium", "High"]
cm = confusion_matrix(y_test, y_pred_class)

print("\n── Confusion matrix ──")
print(pd.DataFrame(cm, index=label_names, columns=label_names))

# ── 12. Visualisations ────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Linear Regression — Student Stress Analysis", fontsize=14, fontweight="bold")

# 12a  Actual vs predicted (continuous)
ax = axes[0, 0]
ax.scatter(y_test, y_pred_cont, alpha=0.5, color="#378ADD", edgecolors="none", s=30)
ax.plot([0, 2], [0, 2], "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlabel("Actual stress (encoded)")
ax.set_ylabel("Predicted stress (continuous)")
ax.set_title("Actual vs. Predicted")
ax.legend(fontsize=9)
ax.set_xlim(-0.2, 2.2)
ax.set_ylim(-0.2, 2.5)

# 12b  Residuals vs fitted
residuals = y_test.values - y_pred_cont
ax = axes[0, 1]
ax.scatter(y_pred_cont, residuals, alpha=0.5, color="#1D9E75", edgecolors="none", s=30)
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs. Fitted")

# 12c  Coefficient bar chart
ax = axes[1, 0]
colors = ["#E24B4A" if c > 0 else "#378ADD" for c in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Standardised coefficient")
ax.set_title("Feature coefficients")

# 12d  Confusion matrix heatmap
ax = axes[1, 1]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion matrix (rounded predictions)")

plt.tight_layout()
plt.savefig("linear_regression_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to linear_regression_plots.png")

# ── 13. Predicted vs actual table (first 15 rows) ─────────────────────────────

label_map = {0: "Low", 1: "Medium", 2: "High"}
results_df = pd.DataFrame({
    "Actual (numeric)":    y_test.values[:15],
    "Actual (label)":      [label_map[v] for v in y_test.values[:15]],
    "Predicted (cont.)":   y_pred_cont[:15].round(3),
    "Predicted (rounded)": [label_map[v] for v in y_pred_class[:15]],
    "Correct":             ["✓" if a == p else "✗"
                             for a, p in zip(y_test.values[:15], y_pred_class[:15])],
})

print("\n── Sample predictions (first 15 test observations) ──")
print(results_df.to_string(index=False))