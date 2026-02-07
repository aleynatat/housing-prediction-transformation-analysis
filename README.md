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

### 4. Key Results & Insights

| Model | Strategy | R2 Score | Insight |
| :--- | :--- | :--- | :--- |
| **XGBoost** | Base (No Trans/Tuning) | 0.8297 | Solid baseline performance. |
| **LightGBM** | Only Transformation | 0.8306 | Minimal gain over baseline. Transformation alone wasn't enough. |
| **LightGBM** | Only Tuning (Raw Data) | 0.8409 | Significant improvement. Proved that optimization is crucial. |
| **LightGBM** | **Tuning + Transformation** | **0.8444** 🏆 | **Best Performance.** Combining both techniques unlocked the highest accuracy. |

### 5. Critical Finding

The analysis revealed a synergistic effect between **Feature Transformation** and **Hyperparameter Tuning**. While Tuning alone provided a strong boost (R2: 0.8409), applying **Power Transformations (Yeo-Johnson & Box-Cox)** before tuning allowed the model to reach its peak performance (**R2: 0.8444**).

This contradicts the initial hypothesis that tree-based models don't benefit from transformations. In this dataset, normalizing the distributions likely provided a smoother optimization surface, enabling `RandomizedSearchCV` to converge on a superior set of hyperparameters.
