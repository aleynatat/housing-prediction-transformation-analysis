# housing-prediction-transformation-analysis

# Impact of Feature Transformation on Housing Price Prediction

This project analyzes the California Housing dataset to predict median house values. The core objective is to benchmark **Tree-Based Models (XGBoost & LightGBM)** and investigate whether **Power Transformations (Yeo-Johnson)** improve model performance compared to **Hyperparameter Tuning**.

##  Project Goals
1.  **Benchmark Models:** Compare the performance of XGBoost and LightGBM.
2.  **Analyze Feature Transformation:** Test if applying `PowerTransformer` (Yeo-Johnson) to input features improves R2 scores for tree-based models.
3.  **Optimize Performance:** Use `RandomizedSearchCV` to find the best hyperparameters.

##  Methodology

### 1. Data Preprocessing
* **Missing Values:** Imputed `total_bedrooms` using the median strategy (to prevent data leakage, imputation is handled carefully).
* **Categorical Encoding:** Applied `OneHotEncoder` to the `ocean_proximity` feature.
* **Transformation:** Applied **Yeo-Johnson Power Transformation** to normalize feature distributions.

### 2. Models Implemented
* **XGBoost Regressor** (Base & Tuned)
* **LightGBM Regressor** (Base & Tuned)

### 3. Evaluation Metrics
* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R2 Score

##  Key Results & Insights

| Model | Strategy | R2 Score | Insight |
|-------|----------|----------|---------|
| XGBoost | Base | ~0.83 | Good baseline performance. |
| **LightGBM** | **Tuned** | **~0.84** | **Best Performance.** Tuning proved more effective than transformation. |
| XGBoost | Transformed (Yeo-Johnson) | ~0.83 | Transformation had negligible impact. |
| LightGBM | Transformed (Yeo-Johnson) | ~0.83 | Slightly lower performance than the tuned model. |

###  Critical Finding
The analysis demonstrates that for tree-based algorithms (which split data based on thresholds rather than distribution), **Hyperparameter Tuning yielded better results than Feature Transformation**. While Power Transformations (like Yeo-Johnson) are crucial for linear models, they provided minimal gain for XGBoost/LightGBM in this dataset.
