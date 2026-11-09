import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error
)

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the merged dataset
df = pd.read_csv(
    "Data/processed/merged.csv",
    parse_dates=["date"]
)

# Display basic information
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nSales statistics:")
print(df["sales"].describe())
# 1. Check all column names and data types

print("\n--- COLUMN INFORMATION ---")

for column in df.columns:
    print(column, ":", df[column].dtype)


# 2. Check missing values

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())


# 3. Check sales statistics

print("\n--- SALES STATISTICS ---")
print(df["sales"].describe())


# 4. Check date range

print("\n--- DATE RANGE ---")
print("First date:", df["date"].min())
print("Last date:", df["date"].max())


# 5. Check categorical variables

print("\n--- UNIQUE VALUES ---")
print("Number of product families:", df["family"].nunique())
print("Number of stores:", df["store_nbr"].nunique())
print("Number of cities:", df["city"].nunique())
# ==========================================
# STEP 2: TRAIN-TEST SPLIT
# ==========================================

# Sort the dataset by date
df = df.sort_values("date")

# Choose the cutoff date
split_date = "2017-05-01"

# Training data: before May 2017
train = df[df["date"] < split_date].copy()

# Testing data: May 2017 onwards
test = df[df["date"] >= split_date].copy()

# Display the results
print("\n--- TRAINING AND TESTING DATA ---")

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

print("Training date range:",
      train["date"].min(), "to", train["date"].max())

print("Testing date range:",
      test["date"].min(), "to", test["date"].max())
# ==========================================
# STEP 3: SELECT FEATURES
# ==========================================

# Columns that describe the store and product
categorical_features = [
    "store_nbr",
    "family",
    "city",
    "state",
    "store_type",
    "season"
]

# Numerical features (without weather)
numerical_features = [
    "onpromotion",
    "cluster",
    "oil_price",
    "is_holiday",
    "year",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "is_weekend",
    "is_payday"
]

# Weather features
weather_features = [
    "temperature_2m_mean",
    "precipitation_sum",
    "is_rainy"
]

# Version A: Without weather
features_without_weather = (
    categorical_features + numerical_features
)

# Version B: With weather
features_with_weather = (
    features_without_weather + weather_features
)

# Target variable
target = "sales"

print("\n--- FEATURE SELECTION ---")

print("Number of features without weather:",
      len(features_without_weather))

print("Number of features with weather:",
      len(features_with_weather))

print("\nFeatures without weather:")
print(features_without_weather)

print("\nFeatures with weather:")
print(features_with_weather)
# ==========================================
# STEP 4: ENCODE CATEGORICAL FEATURES
# ==========================================

# Create the preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

print("\n--- PREPROCESSOR CREATED ---")
print("Categorical features will be encoded.")
print("Numerical features will remain unchanged.")
# ==========================================
# STEP 5: CREATE X AND y
# ==========================================

# Training and testing targets
y_train = np.log1p(train[target])
y_test = np.log1p(test[target])

# Features without weather
X_train_without_weather = train[features_without_weather]
X_test_without_weather = test[features_without_weather]

# Features with weather
X_train_with_weather = train[features_with_weather]
X_test_with_weather = test[features_with_weather]

print("\n--- DATA PREPARATION ---")

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("Training features without weather:",
      X_train_without_weather.shape)

print("Testing features without weather:",
      X_test_without_weather.shape)

print("Training features with weather:",
      X_train_with_weather.shape)

print("Testing features with weather:",
      X_test_with_weather.shape)
# BASELINE MODEL

baseline_model = DummyRegressor(strategy="mean")

# Train the model
baseline_model.fit(
    np.zeros((len(y_train), 1)),
    y_train
)

# Make predictions
baseline_predictions_log = baseline_model.predict(
    np.zeros((len(y_test), 1))
)

# Convert predictions back to original sales scale
baseline_predictions = np.expm1(baseline_predictions_log)

# Actual sales values
y_test_actual = test["sales"].to_numpy()

print("\n--- BASELINE MODEL ---")
print("Baseline training completed!")
# EVALUATE BASELINE

baseline_mae = mean_absolute_error(
    y_test_actual,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test_actual, baseline_predictions)
)

baseline_r2 = r2_score(
    y_test_actual,
    baseline_predictions
)

baseline_rmsle = np.sqrt(
    mean_squared_log_error(
        y_test_actual,
        np.maximum(baseline_predictions, 0)
    )
)

print("\n--- BASELINE RESULTS ---")
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
print("R2:", baseline_r2)
print("RMSLE:", baseline_rmsle)
# DECISION TREE - WITHOUT WEATHER

# Take a sample from the training data
tree_train = train.sample(
    n=300_000,
    random_state=42
)

# Select features and target
X_tree_train = tree_train[features_without_weather]
y_tree_train = np.log1p(tree_train[target])

print("\n--- DECISION TREE DATA ---")
print("Training sample:", X_tree_train.shape)
print("Testing data:", X_test_without_weather.shape)
# Create the Decision Tree pipeline

decision_tree = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=10,
        random_state=42
    ))
])

print("\nDecision Tree created successfully!")
print("\nTraining Decision Tree...")

decision_tree.fit(
    X_tree_train,
    y_tree_train
)

print("Decision Tree training completed!")
# Make predictions on the test data

tree_predictions_log = decision_tree.predict(
    X_test_without_weather
)

# Convert predictions back to original sales scale
tree_predictions = np.expm1(tree_predictions_log)

print("Predictions completed!")
# Evaluate Decision Tree

tree_mae = mean_absolute_error(
    y_test_actual,
    tree_predictions
)

tree_rmse = np.sqrt(
    mean_squared_error(y_test_actual, tree_predictions)
)

tree_r2 = r2_score(
    y_test_actual,
    tree_predictions
)

tree_rmsle = np.sqrt(
    mean_squared_log_error(
        y_test_actual,
        np.maximum(tree_predictions, 0)
    )
)

print("\n--- DECISION TREE RESULTS (WITHOUT WEATHER) ---")
print("MAE:", tree_mae)
print("RMSE:", tree_rmse)
print("R2:", tree_r2)
print("RMSLE:", tree_rmsle)
# DECISION TREE - WITH WEATHER

X_tree_train_weather = tree_train[features_with_weather]

print("\n--- DECISION TREE WITH WEATHER DATA ---")
print("Training sample:", X_tree_train_weather.shape)
print("Testing data:", X_test_with_weather.shape)
decision_tree_weather = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(
        max_depth=15,
        min_samples_leaf=10,
        random_state=42
    ))
])

print("Decision Tree with weather created!")
print("\nTraining Decision Tree with weather...")

decision_tree_weather.fit(
    X_tree_train_weather,
    y_tree_train
)

print("Training completed!")
tree_weather_predictions_log = decision_tree_weather.predict(
    X_test_with_weather
)

tree_weather_predictions = np.expm1(
    tree_weather_predictions_log
)

print("Weather model predictions completed!")
tree_weather_mae = mean_absolute_error(
    y_test_actual,
    tree_weather_predictions
)

tree_weather_rmse = np.sqrt(
    mean_squared_error(y_test_actual, tree_weather_predictions)
)

tree_weather_r2 = r2_score(
    y_test_actual,
    tree_weather_predictions
)

tree_weather_rmsle = np.sqrt(
    mean_squared_log_error(
        y_test_actual,
        np.maximum(tree_weather_predictions, 0)
    )
)

print("\n--- DECISION TREE RESULTS (WITH WEATHER) ---")
print("MAE:", tree_weather_mae)
print("RMSE:", tree_weather_rmse)
print("R2:", tree_weather_r2)
print("RMSLE:", tree_weather_rmsle)
# RANDOM FOREST - WITHOUT WEATHER

random_forest = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nRandom Forest created successfully!")
print("\nTraining Random Forest...")

random_forest.fit(
    X_tree_train,
    y_tree_train
)

print("Random Forest training completed!")
rf_predictions_log = random_forest.predict(
    X_test_without_weather
)

rf_predictions = np.expm1(rf_predictions_log)

print("Random Forest predictions completed!")
rf_mae = mean_absolute_error(
    y_test_actual,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(y_test_actual, rf_predictions)
)

rf_r2 = r2_score(
    y_test_actual,
    rf_predictions
)

rf_rmsle = np.sqrt(
    mean_squared_log_error(
        y_test_actual,
        np.maximum(rf_predictions, 0)
    )
)

print("\n--- RANDOM FOREST RESULTS (WITHOUT WEATHER) ---")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)
print("RMSLE:", rf_rmsle)
# RANDOM FOREST - WITH WEATHER

random_forest_weather = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nRandom Forest with weather created successfully!")
print("\nTraining Random Forest with weather...")

random_forest_weather.fit(
    X_tree_train_weather,
    y_tree_train
)

print("Random Forest with weather training completed!")
rf_weather_predictions_log = random_forest_weather.predict(
    X_test_with_weather
)

rf_weather_predictions = np.expm1(
    rf_weather_predictions_log
)

print("Random Forest weather predictions completed!")
rf_weather_mae = mean_absolute_error(
    y_test_actual,
    rf_weather_predictions
)

rf_weather_rmse = np.sqrt(
    mean_squared_error(y_test_actual, rf_weather_predictions)
)

rf_weather_r2 = r2_score(
    y_test_actual,
    rf_weather_predictions
)

rf_weather_rmsle = np.sqrt(
    mean_squared_log_error(
        y_test_actual,
        np.maximum(rf_weather_predictions, 0)
    )
)

print("\n--- RANDOM FOREST RESULTS (WITH WEATHER) ---")
print("MAE:", rf_weather_mae)
print("RMSE:", rf_weather_rmse)
print("R2:", rf_weather_r2)
print("RMSLE:", rf_weather_rmsle)


# -----------------------------------------
# FEATURE IMPORTANCE ANALYSIS
# -----------------------------------------

print("\n--- FEATURE IMPORTANCE ANALYSIS ---")

# Use the Random Forest model WITH weather
# so that the feature names and importance scores match.

fitted_preprocessor = random_forest_weather.named_steps["preprocessor"]

fitted_model = random_forest_weather.named_steps["model"]

# Get feature names after preprocessing
feature_names = fitted_preprocessor.get_feature_names_out()

# Get importance scores
importances = fitted_model.feature_importances_

# Check that both have the same length
print("Number of feature names:", len(feature_names))
print("Number of importance scores:", len(importances))

# Create the feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort from most important to least important
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15))

# Save the results
importance_df.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")

# -----------------------------------------
# MODEL COMPARISON TABLE
# -----------------------------------------

print("\n--- MODEL COMPARISON ---")

comparison_df = pd.DataFrame({
    "Model": [
        "Baseline",
        "Decision Tree (Without Weather)",
        "Decision Tree (With Weather)",
        "Random Forest (Without Weather)",
        "Random Forest (With Weather)"
    ],

    "MAE": [
        baseline_mae,
        tree_mae,
        tree_weather_mae,
        rf_mae,
        rf_weather_mae
    ],

    "RMSE": [
        baseline_rmse,
        tree_rmse,
        tree_weather_rmse,
        rf_rmse,
        rf_weather_rmse
    ],

    "R2": [
        baseline_r2,
        tree_r2,
        tree_weather_r2,
        rf_r2,
        rf_weather_r2
    ],

    "RMSLE": [
        baseline_rmsle,
        tree_rmsle,
        tree_weather_rmsle,
        rf_rmsle,
        rf_weather_rmsle
    ]
})

print(comparison_df.round(4))

# Save comparison results
comparison_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")
print("Model comparison saved successfully!")
# ==========================================
# WEATHER FEATURE IMPORTANCE
# ==========================================

# Get feature names from the Random Forest WITH weather
weather_feature_names = (
    random_forest_weather
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

# Get importance scores
weather_importances = (
    random_forest_weather
    .named_steps["model"]
    .feature_importances_
)

# Create a DataFrame
weather_importance_df = pd.DataFrame({
    "Feature": weather_feature_names,
    "Importance": weather_importances
})

# Sort features by importance
weather_importance_df = weather_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display weather features only
print("\n--- WEATHER FEATURE IMPORTANCE ---")

weather_only = weather_importance_df[
    weather_importance_df["Feature"].str.contains(
        "temperature|precipitation|is_rainy",
        case=False
    )
]

print(weather_only)

# Save the results
weather_importance_df.to_csv(
    "results/feature_importance_with_weather.csv",
    index=False
)

print("\nWeather feature importance saved!")