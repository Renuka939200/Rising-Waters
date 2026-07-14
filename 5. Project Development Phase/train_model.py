import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# -----------------------------
# Create folders if not present
# -----------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/flood.csv")

print("\nDataset Loaded Successfully")
print(df.head())

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("Flood", axis=1)
y = df["Flood"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

print("Scaler Saved")

# -----------------------------
# Models
# -----------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
}

accuracies = {}

best_model = None
best_name = ""
best_accuracy = 0

# -----------------------------
# Train Models
# -----------------------------
print("\nTraining Models...\n")

for name, model in models.items():

    if name == "KNN":
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)

    else:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    accuracies[name] = accuracy

    print(f"{name} Accuracy : {accuracy*100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name
        best_prediction = prediction

# -----------------------------
# Save Best Model
# -----------------------------
joblib.dump(best_model, "model/best_model.pkl")

print("\nBest Model :", best_name)
print("Accuracy :", round(best_accuracy*100,2),"%")
print("Model Saved Successfully!")

# -----------------------------
# Accuracy Graph
# -----------------------------
plt.figure(figsize=(8,5))

plt.bar(
    accuracies.keys(),
    accuracies.values()
)

plt.ylabel("Accuracy")

plt.title("Model Comparison")

plt.ylim(0,1)

plt.savefig("static/graphs/accuracy.png")

plt.close()

print("Accuracy Graph Saved")

# -----------------------------
# Confusion Matrix
# -----------------------------
ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_prediction
)

plt.title(best_name + " Confusion Matrix")

plt.savefig("static/graphs/confusion_matrix.png")

plt.close()

print("Confusion Matrix Saved")

# -----------------------------
# Feature Importance
# -----------------------------
if best_name in ["Decision Tree","Random Forest","XGBoost"]:

    importance = best_model.feature_importances_

    plt.figure(figsize=(9,5))

    plt.bar(
        X.columns,
        importance
    )

    plt.xticks(rotation=20)

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig("static/graphs/feature_importance.png")

    plt.close()

    print("Feature Importance Saved")

print("\nProject Training Completed Successfully!")