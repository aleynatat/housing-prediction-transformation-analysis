import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA LOADING & EXPLORATION
# ==========================================
df = pd.read_csv('21-housing.csv')

print("--- Data Overview ---")
print(df.head())
print(df.describe())
print(df.shape)
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Info ---")
print(df.info())

# ==========================================
# 2. VISUALIZATION
# ==========================================
def plot_all_histograms(df, title_prefix=""):
    num_cols = df.select_dtypes(include=np.number).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    plt.figure(figsize=(n_cols * 3, n_rows * 3))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(title_prefix + col)
        plt.xlabel("")
        plt.ylabel("")
    plt.tight_layout()
    plt.show()

plot_all_histograms(df, "Original - ")

# Correlation Matrix
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# ==========================================
# 3. OUTLIER DETECTION (Functions Defined)
# ==========================================
def find_outliers_iqr(df, threshold=1.5):
    outlier_summary = {}
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        outlier_summary[col] = {
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(df) * 100,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    return pd.DataFrame(outlier_summary)

print("\n--- Outlier Summary ---")
print(find_outliers_iqr(df, threshold=1.5))

# Note: We are deliberately skipping outlier removal to preserve valuable data points
# (e.g., high-value houses) for the tree-based models.
print("Original Data Shape:", df.shape)
df_clean = df.copy()

# ==========================================
# 4. PREPROCESSING (Encoding)
# ==========================================
print("\n--- Categorical Distribution ---")
print(df_clean["ocean_proximity"].value_counts())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_cols = ["ocean_proximity"]

# Using set_output(transform="pandas") to keep the result as a DataFrame
preprocessor = ColumnTransformer(
    transformers=[
        # sparse_output=False is crucial to avoid sparse matrix errors
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_cols),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False  # Keeps original column names clean
).set_output(transform="pandas")

df_encoded = preprocessor.fit_transform(df_clean)

print("\n--- Encoded Dataframe ---")
print(df_encoded.head())
print(df_encoded.info())

# ==========================================
# 5. TRAIN-TEST SPLIT & IMPUTATION
# ==========================================
X = df_encoded.drop("median_house_value", axis=1)
y = df_encoded["median_house_value"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

# Imputing missing values AFTER split to prevent data leakage
train_median = X_train["total_bedrooms"].median()

X_train["total_bedrooms"] = X_train["total_bedrooms"].fillna(train_median)
X_test["total_bedrooms"] = X_test["total_bedrooms"].fillna(train_median)

# ==========================================
# 6. MODELING & EVALUATION
# ==========================================
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(true, predicted):
    mse = mean_squared_error(true, predicted)
    r2 = r2_score(true, predicted)
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mse)
    return (f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2 Score: {r2:.4f}")

# --- 6.1 XGBoost (Base Model) ---
from xgboost import XGBRegressor
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n----- XGBoost Without Transformation -----")
print(evaluate_model(y_test, y_pred_xgb))

# --- 6.2 XGBoost (Hyperparameter Tuning) ---
xgboost_params = {
    "learning_rate": [0.01, 0.1, 0.2, 0.5],
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    "n_estimators": [100, 200, 300, 400, 500],
    "colsample_bytree": [0.2, 0.3, 0.4, 0.5, 0.7, 1],
}

from sklearn.model_selection import RandomizedSearchCV
randomcv = RandomizedSearchCV(estimator=XGBRegressor(random_state=42),
                              param_distributions=xgboost_params,
                              cv=5, n_jobs=-1, verbose=1)
randomcv.fit(X_train, y_train)

print(f"\nBest XGBoost Params: {randomcv.best_params_}")

y_pred_xgb_tuned = randomcv.predict(X_test)
print("\n----- XGBoost With Hyperparameter Tuning -----")
print(evaluate_model(y_test, y_pred_xgb_tuned))

# --- 6.3 LightGBM (Base Model) ---
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)

y_pred_lgbm = lgbm.predict(X_test)
print("\n----- LightGBM Without Transformation -----")
print(evaluate_model(y_test, y_pred_lgbm))

# --- 6.4 LightGBM (Hyperparameter Tuning) ---
lgbm_params = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3],
    'max_depth': [3, 4, 5, 6, 7, -1],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.6, 0.8, 1],
    'colsample_bytree': [0.6, 0.8, 1],
    "n_estimators": [100, 200, 300, 400, 500, 1000],
    "num_leaves": [15, 32, 63, 127]
}

randomcv2 = RandomizedSearchCV(estimator=LGBMRegressor(verbose=-1, random_state=42),
                               param_distributions=lgbm_params,
                               cv=5, n_jobs=-1, verbose=1)
randomcv2.fit(X_train, y_train)

print(f"\nBest LightGBM Params: {randomcv2.best_params_}")

y_pred_lgbm_tuned = randomcv2.predict(X_test)
print("\n----- LightGBM With Hyperparameter Tuning -----")
print(evaluate_model(y_test, y_pred_lgbm_tuned))

# ==========================================
# 7. FEATURE TRANSFORMATION (Yeo-Johnson)
# ==========================================
from sklearn.preprocessing import PowerTransformer
power_transformer = PowerTransformer(method='yeo-johnson')

# Applying transformation ONLY to X features
X_train_transformed = power_transformer.fit_transform(X_train)
X_test_transformed = power_transformer.transform(X_test)

# Converting back to DataFrame for better handling
X_columns = power_transformer.get_feature_names_out()
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=X_columns)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=X_columns)

plot_all_histograms(X_train_transformed_df, "Transformed - ")

# --- XGBoost with Transformation ---
xgb_transformed = XGBRegressor(random_state=42)
xgb_transformed.fit(X_train_transformed_df, y_train)

y_pred_xgb_trans = xgb_transformed.predict(X_test_transformed_df)

print("\n----- XGBoost With Transformation (Only X) -----")
print(evaluate_model(y_test, y_pred_xgb_trans))

# --- LightGBM with Transformation ---
lgbm_transformed = LGBMRegressor(verbose=-1, random_state=42)
lgbm_transformed.fit(X_train_transformed_df, y_train)

y_pred_lgbm_trans = lgbm_transformed.predict(X_test_transformed_df)

print("\n----- LightGBM With Transformation (Only X) -----")
print(evaluate_model(y_test, y_pred_lgbm_trans))

