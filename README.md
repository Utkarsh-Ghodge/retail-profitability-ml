# Retail Order Profitability & Customer Segmentation

An end-to-end machine learning pipeline analyzing retail order transactions (`Global_Superstore_ug_050.csv`, $N = 1,000$) to segment customer purchasing behavior and predict order-level profitability.

## Project Pipeline

1. **Data Preprocessing & Feature Engineering**
   - **Missing Data Treatment:** Dropped `Postal Code` (80.6% missing) and 8 severely incomplete records (>30% missing), retaining 992 high-integrity orders. Applied mode imputation for categorical attributes and median imputation for continuous ratio metrics.
   - **Encoding & Scaling:** Encoded 10 operational and geographic categorical features via `OrdinalEncoder`. Applied IQR-based 5% two-sided Winsorization to `Sales`, `Discount`, `Profit`, and `Shipping Cost` to tame extreme right-skewed tails before Min-Max scaling to `[0, 1]`.

2. **Unsupervised Learning (Segmentation & Latent Structure)**
   - **K-Means Clustering ($K=4$):** Selected $K=4$ via peak Silhouette Score (`0.2800`) and lowest Davies-Bouldin Index (`1.1953`), validated by one-way ANOVA ($p < 0.0001$ across all numeric features):
     - *Mid-Tier Profitable Orders ($n=470$):* $1,539.27 avg. sales, 3% avg. discount, $246.83 avg. profit.
     - *High-Value Bulk Orders ($n=226$):* $2,975.65 avg. sales, 3% avg. discount, $777.24 avg. profit.
     - *Standard Low-Ticket Orders ($n=183$):* $292.93 avg. sales, 10% avg. discount, $33.38 avg. profit (concentrated in USCA).
     - *Discount-Driven Loss Orders ($n=113$):* $1,611.52 avg. sales eroded by 33% avg. discount, resulting in -$104.23 avg. net loss.
   - **PCA / Factor Analysis (Varimax Rotation):** Extracted 2 orthogonal factors (eigenvalues `2.4751` and `1.1871`, explaining 73.24% variance) after removing cross-loaded `Profit`: **Factor 1 (Order Volume & Revenue Scale)** and **Factor 2 (Promotional Discount Intensity)**.

3. **Supervised & Ensemble Profitability Classification**
   - **Leakage-Free Setup:** Predicted binary profitability (`Is_Profitable`: 82.46% profitable vs. 17.54% loss-making) using 14 predictors (`Profit` excluded) with an 80/20 stratified split (`random_state=75050`) and 10-sample bootstrapping for ensemble stability.

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8744** | **0.8883** | 0.9695 | **0.9271** | **0.8720** |
| **Decision Tree** (`max_depth=5`) | 0.8643 | 0.8870 | 0.9573 | 0.9208 | 0.8699 |
| **Random Forest** (Bagging, 10-Sample Avg) | 0.8638 | 0.8769 | **0.9722** | 0.9220 | 0.8209 |
| **XGBoost** (Boosting, 10-Sample Avg) | 0.8367 | 0.8868 | 0.9206 | 0.9032 | 0.8322 |

## Key Business Takeaways

- **Discount Governance:** Promotional discounting acts independently of order volume (Factor 2) and is the primary driver of margin erosion—orders averaging 33% discounts turn net-negative despite high top-line revenue.
- **Model Selection:** **Logistic Regression** achieves the highest overall accuracy (`87.44%`), F1-score (`0.9271`), and ROC-AUC (`0.8720`) due to strong monotonic relationships between discount depth and profitability, while **Random Forest** provides the strongest bootstrapped recall (`97.22%`).

## Repository Contents

- [`retail_profitability_analysis.ipynb`](retail_profitability_analysis.ipynb) — Complete Python notebook with visualizations, statistical tests, and academic inferences.
- [`Global_Superstore_ug_050.csv`](Global_Superstore_ug_050.csv) — Retail order dataset (1,000 records, 24 features).

## Tech Stack

`Python` • `Pandas` • `NumPy` • `Scikit-Learn` • `XGBoost` • `SciPy` • `Matplotlib` • `Seaborn`
