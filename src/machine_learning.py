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

import matplotlib.pyplot as plt


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
#        TRAIN-TEST SPLIT
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
#   SELECT FEATURES
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
#      ENCODE CATEGORICAL FEATURES
# ==========================================

# Create the preprocessing transformer


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(
                handle_unknown="ignore"), categorical_features)
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

# -----------------------------------------
# BASELINE MODELS
# -----------------------------------------

# Actual sales values
y_test_actual = test["sales"].to_numpy()


# =========================================
# BASELINE 1: SIMPLE AVERAGE
# =========================================

print("\n--- BASELINE 1: SIMPLE AVERAGE ---")

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

# Evaluate
baseline_mae = mean_absolute_error(
    y_test_actual, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test_actual, baseline_predictions)
)

baseline_r2 = r2_score(
    y_test_actual, baseline_predictions
)

baseline_rmsle = np.sqrt(
    mean_squared_log_error(
        y_test_actual,
        np.maximum(baseline_predictions, 0)
    )
)

print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
print("R2:", baseline_r2)
print("RMSLE:", baseline_rmsle)


# =========================================
# BASELINE 2: STORE-FAMILY AVERAGE
# =========================================

print("\n--- BASELINE 2: STORE-FAMILY AVERAGE ---")

# Calculate average sales for each store and product family
# using TRAINING DATA ONLY
store_family_means = train.groupby(
    ["store_nbr", "family"]
)["sales"].mean()

# Calculate the overall training average
global_mean = train["sales"].mean()

# Match each test row with its store-family average
store_family_predictions = test[
    ["store_nbr", "family"]
].merge(
    store_family_means.rename("prediction").reset_index(),
    on=["store_nbr", "family"],
    how="left"
)["prediction"]

# Use global average if a store-family combination is unseen
store_family_predictions = store_family_predictions.fillna(
    global_mean
).to_numpy()

# Evaluate
store_family_mae = mean_absolute_error(
    y_test_actual, store_family_predictions
)

store_family_rmse = np.sqrt(
    mean_squared_error(y_test_actual, store_family_predictions)
)

store_family_r2 = r2_score(
    y_test_actual, store_family_predictions
)

store_family_rmsle = np.sqrt(
    mean_squared_log_error(
        y_test_actual,
        np.maximum(store_family_predictions, 0)
    )
)

print("MAE:", store_family_mae)
print("RMSE:", store_family_rmse)
print("R2:", store_family_r2)
print("RMSLE:", store_family_rmsle)

print("\nBoth baseline models completed successfully!")

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
    ("preprocessor", make_preprocessor()),
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
    ("preprocessor", make_preprocessor()),
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
    ("preprocessor", make_preprocessor()),
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
    ("preprocessor", make_preprocessor()),
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
# ERROR ANALYSIS
# -----------------------------------------

print("\n--- ERROR ANALYSIS ---")

# Create a DataFrame containing actual and predicted sales
error_analysis_df = test[
    ["store_nbr", "family", "sales"]
].copy()

error_analysis_df["predicted_sales"] = rf_predictions

# Calculate absolute prediction error
error_analysis_df["absolute_error"] = abs(
    error_analysis_df["sales"] -
    error_analysis_df["predicted_sales"]
)


# =========================================
# 1. ERROR BY PRODUCT FAMILY
# =========================================

family_errors = error_analysis_df.groupby(
    "family"
)["absolute_error"].mean().sort_values(
    ascending=False
)

print("\nTop 10 Product Families with Highest MAE:")
print(family_errors.head(10))


# =========================================
# 2. ERROR BY STORE
# =========================================

store_errors = error_analysis_df.groupby(
    "store_nbr"
)["absolute_error"].mean().sort_values(
    ascending=False
)

print("\nTop 10 Stores with Highest MAE:")
print(store_errors.head(10))


# Save error analysis results
family_errors.to_csv(
    "results/error_by_family.csv"
)

store_errors.to_csv(
    "results/error_by_store.csv"
)

print("\nError analysis files saved successfully!")


# =========================================
# 3. NORMALIZED MAE BY PRODUCT FAMILY
# =========================================

print("\n--- NORMALIZED MAE BY PRODUCT FAMILY ---")

# Calculate average actual sales and MAE for each family
family_error_summary = error_analysis_df.groupby(
    "family"
).agg(
    mean_actual_sales=("sales", "mean"),
    MAE=("absolute_error", "mean")
).reset_index()

# Calculate error relative to average sales volume
family_error_summary["Normalized_MAE_%"] = (
    family_error_summary["MAE"] /
    family_error_summary["mean_actual_sales"]
) * 100

# Sort from highest relative error to lowest
family_error_summary = family_error_summary.sort_values(
    by="Normalized_MAE_%",
    ascending=False
)

print("\nProduct Families with Highest Relative Errors:")
print(family_error_summary.head(15).round(2))

# Save results
family_error_summary.to_csv(
    "results/normalized_error_by_family.csv",
    index=False
)

print("\nNormalized family error analysis saved successfully!")


# =========================================
# 4. FILTER LOW-VOLUME PRODUCT FAMILIES
# =========================================

print("\n--- NORMALIZED MAE (AVERAGE SALES >= 10) ---")

# Keep product families with average sales of at least 10
filtered_family_errors = family_error_summary[
    family_error_summary["mean_actual_sales"] >= 10
].copy()

# Sort by relative error
filtered_family_errors = filtered_family_errors.sort_values(
    by="Normalized_MAE_%",
    ascending=False
)

print(filtered_family_errors.round(2))

# Save filtered results
filtered_family_errors.to_csv(
    "results/normalized_error_by_family_filtered.csv",
    index=False
)

print("\nFiltered family error analysis saved successfully!")


# -----------------------------------------
# FEATURE IMPORTANCE ANALYSIS
# -----------------------------------------

print("\n--- FEATURE IMPORTANCE ANALYSIS ---")


# FUNCTION TO EXTRACT AND SAVE FEATURE IMPORTANCE
def save_feature_importance(model_pipeline, filename, model_name):

    fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
    fitted_model = model_pipeline.named_steps["model"]

    # Get feature names after preprocessing
    feature_names = fitted_preprocessor.get_feature_names_out()

    # Get importance scores
    importances = fitted_model.feature_importances_

    # Check that both have the same length
    print(f"\n{model_name}")
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
    importance_df.to_csv(filename, index=False)

    print(f"\nSaved successfully: {filename}")


# 1. Random Forest WITHOUT weather
save_feature_importance(
    random_forest,
    "results/feature_importance.csv",
    "Random Forest WITHOUT Weather"
)


# 2. Random Forest WITH weather
save_feature_importance(
    random_forest_weather,
    "results/feature_importance_with_weather.csv",
    "Random Forest WITH Weather"
)

# -----------------------------------------
# MODEL COMPARISON TABLE
# -----------------------------------------

print("\n--- MODEL COMPARISON ---")

comparison_df = pd.DataFrame({
    "Model": [
        "Simple Average Baseline",
        "Store-Family Average Baseline",
        "Decision Tree (Without Weather)",
        "Decision Tree (With Weather)",
        "Random Forest (Without Weather)",
        "Random Forest (With Weather)"
    ],

    "MAE": [
        baseline_mae,
        store_family_mae,
        tree_mae,
        tree_weather_mae,
        rf_mae,
        rf_weather_mae
    ],

    "RMSE": [
        baseline_rmse,
        store_family_rmse,
        tree_rmse,
        tree_weather_rmse,
        rf_rmse,
        rf_weather_rmse
    ],

    "R2": [
        baseline_r2,
        store_family_r2,
        tree_r2,
        tree_weather_r2,
        rf_r2,
        rf_weather_r2
    ],

    "RMSLE": [
        baseline_rmsle,
        store_family_rmsle,
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
# ==========================================
# VISUALIZATION 1: MODEL COMPARISON (MAE)
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    comparison_df["Model"],
    comparison_df["MAE"]
)

plt.title("Model Comparison - MAE")
plt.xlabel("Model")
plt.ylabel("MAE")

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    "figures/model_comparison_mae.png",
    dpi=300
)

plt.close()

print("MAE comparison graph saved successfully!")
# ==========================================
#      CREATE REMAINING COMPARISON GRAPHS
# ==========================================

# Load model comparison results
comparison_df = pd.read_csv("results/model_comparison.csv")

# Metrics to visualize
metrics = ["RMSE", "R2", "RMSLE"]

for metric in metrics:

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["Model"],
        comparison_df[metric]
    )

    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()

    plt.savefig(f"figures/model_comparison_{metric.lower()}.png")
    plt.close()

print("All comparison graphs saved successfully!")

# ==========================================
# ACTUAL VS PREDICTED SALES
# ==========================================

print("\n--- ACTUAL VS PREDICTED SALES GRAPH ---")

# Select 5,000 random test observations
sample_indices = np.random.RandomState(42).choice(
    len(y_test_actual),
    size=min(5000, len(y_test_actual)),
    replace=False
)

# Get actual and predicted sales
actual_sample = y_test_actual[sample_indices]
predicted_sample = rf_predictions[sample_indices]

# Apply log1p to make the graph easier to read
actual_log = np.log1p(actual_sample)
predicted_log = np.log1p(predicted_sample)

# Create the graph
plt.figure(figsize=(10, 6))

plt.scatter(
    actual_log,
    predicted_log,
    alpha=0.3,
    s=15
)

# Add the perfect prediction line
max_value = max(actual_log.max(), predicted_log.max())

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.title("Actual vs Predicted Sales - Random Forest")
plt.xlabel("Actual Sales (log1p scale)")
plt.ylabel("Predicted Sales (log1p scale)")

plt.tight_layout()

# Save the graph
plt.savefig(
    "figures/actual_vs_predicted.png",
    dpi=300
)

plt.close()

print("Actual vs Predicted graph saved successfully!")
