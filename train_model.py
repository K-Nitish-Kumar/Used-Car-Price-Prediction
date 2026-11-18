import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------
# 1. CREATE REQUIRED DIRECTORIES
# -----------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------------
# 2. LOAD DATASET
# -----------------------------------

df = pd.read_csv("dataset/car_details.csv")

print("Original dataset shape:", df.shape)

# Normalize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

# Handle different Kaggle CSV versions
df = df.rename(columns={
    "car_name": "name",
    "kms_driven": "km_driven",
    "fuel_type": "fuel",
    "seller": "seller_type",
    "owner": "owner"
})

# -----------------------------------
# 3. SELECT FEATURES
# -----------------------------------

required_columns = [
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}. "
        "Check your Kaggle CSV file."
    )

# Include car model when available
features = [
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

if "name" in df.columns:
    features.append("name")

target = "selling_price"

df = df[features + [target]].copy()

# Convert numeric columns
for col in ["year", "km_driven", target]:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", ""),
        errors="coerce"
    )

# Remove invalid target records
df = df.dropna(subset=[target])
df = df[df[target] > 0]

# Remove duplicate records
df = df.drop_duplicates()

# Normalize text columns
categorical_features = [
    col for col in features
    if col not in ["year", "km_driven"]
]

for col in categorical_features:
    df[col] = df[col].apply(
        lambda x: str(x).strip()
        if pd.notna(x) else np.nan
    )

print("\nCleaned dataset shape:", df.shape)

# -----------------------------------
# 4. DEFINE INPUT AND OUTPUT
# -----------------------------------

X = df[features]
y = df[target]

numeric_features = ["year", "km_driven"]

# -----------------------------------
# 5. PREPROCESSING
# -----------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer,
         categorical_features)
    ]
)

# -----------------------------------
# 6. TRAIN-TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# -----------------------------------
# 7. DEFINE REGRESSION MODELS
# -----------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=2,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

# -----------------------------------
# 8. TRAIN AND EVALUATE
# -----------------------------------

for name, algorithm in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test, predictions
    )

    mse = mean_squared_error(
        y_test, predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test, predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# -----------------------------------
# 9. COMPARE MODELS
# -----------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# -----------------------------------
# 10. SELECT BEST MODEL
# -----------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

# -----------------------------------
# 11. SAVE TRAINED MODEL
# -----------------------------------

model_package = {
    "model": best_model,
    "features": features,
    "best_model_name": best_model_name
}

joblib.dump(
    model_package,
    "models/best_model.pkl"
)

print("\nModel saved successfully!")

# -----------------------------------
# 12. ACTUAL VS PREDICTED GRAPH
# -----------------------------------

best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.6
)

min_value = min(y_test.min(), best_predictions.min())
max_value = max(y_test.max(), best_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png"
)

plt.show()

print("\nMODEL TRAINING COMPLETED!")
