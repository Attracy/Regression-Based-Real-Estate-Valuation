# Regression-Based Real Estate Valuation

A supervised regression project forecasting residential property prices from size, location, and condition features, comparing six regression algorithms and validating a CatBoost final model against classical linear regression assumptions.

---

## Objective

Predict residential property price from structural and location features (size, bedrooms/bathrooms, condition, zip code, house age, waterfront/view), and identify which features drive value most.

## Dataset

- 4,600 residential property records, 18 original columns
- No missing values, no duplicate rows
- Target: `price`
- Dropped as non-predictive/redundant: `date`, `country`, `street`, `statezip` (kept only the extracted `zip`), `city`

## Pipeline

### Data Cleaning
Shape, nulls, duplicates, and dtypes checked upfront — dataset came in clean (0 nulls, 0 duplicates). Zip code extracted from `statezip` since raw city/street/zip strings carry no usable numeric signal on their own.

### Exploratory Data Analysis
Distribution, box, and cumulative plots for every numerical feature (price, sqft_living, sqft_lot, sqft_above, sqft_basement); bar/pie plots for categoricals (bedrooms, condition, floors, waterfront, view, year built/renovated, zip). Full correlation heatmap computed before and after feature engineering to track how engineered features relate to price and to each other.

### Outlier Detection & Removal
Quantile filtering on `price` — bottom 1.5% and top 1.5% removed (3% total), taking the dataset from 4,600 to 4,462 rows. A tail-trim rather than a hard IQR cutoff, chosen to remove extreme listing-price errors without discarding legitimately expensive/cheap properties.

### Feature Engineering
- `total_sqft` = `sqft_basement` + `sqft_living` (finished + basement space combined)
- `waterfront_view` = `waterfront` × `view` (interaction term — a view only commands a premium in combination with waterfront access, or vice versa)
- `zip_tier` — zip codes bucketed into Low/Medium/High tertiles by average price per zip, since raw zip code has no ordinal meaning for a model
- `bedroom_group`, `bathroom_group`, `condition_group` — count/rating features binned into small/medium/large-style categories to reduce noise from sparse exact-count categories
- `house_age` (2025 − `yr_built`) binned into `house_age_tier` (New / Modern / Mid-Age / Old / Very Old)
- Log-transform (`log1p`) applied to `price`, `total_sqft`, `sqft_lot`, `sqft_above` to reduce right-skew
- All engineered categorical tiers ordinal-encoded

### Feature Selection
Three independent statsmodels-based selection methods run and cross-checked against each other: forward selection, backward elimination, and stepwise selection (each using AIC or p-value < 0.05 as the inclusion/exclusion criterion). Raw columns superseded by their engineered equivalents (`bedrooms`, `bathrooms`, `waterfront`, `view`, `condition`, `sqft_basement`, `sqft_living`, `house_age`, `yr_built`, `yr_renovated`, `zip`) were dropped in favor of the engineered groupings, keeping the final feature set compact and non-redundant.

### Univariate Regression Analysis
Individual OLS fits (`sqft_lot`, `total_sqft` against `price`) with scatter + fitted-line plots and residual inspection, run before multivariate modeling to sanity-check the basic price relationships.

### Model Training & Comparison
All models trained on the same 80/20 split (`random_state=42`) with `MinMaxScaler`-scaled features and evaluated via R², Adjusted R², MSE, and 5-fold cross-validated R²:

| Model | R² | Adjusted R² | MSE | CV R² |
|---|---|---|---|---|
| Linear Regression | 0.5575 | 0.5525 | 0.11 | 53.44% |
| Decision Tree (max_depth=5) | 0.7402 | 0.7372 | 0.07 | 69.95% |
| Gradient Boosting | 0.7829 | 0.7805 | 0.06 | 75.19% |
| XGBoost | 0.7751 | 0.7725 | 0.06 | 74.37% |
| Random Forest | 0.7745 | 0.7720 | 0.06 | 72.93% |
| CatBoost (default) | 0.7864 | 0.7840 | 0.05 | 74.99% |

CatBoost and Gradient Boosting are the top two performers; Linear Regression trails well behind, indicating meaningful non-linearity in the price relationships that a linear model can't capture on its own.

### Hyperparameter Tuning — CatBoost
`RandomizedSearchCV` (5-fold, scoring=R², 10 iterations) over iterations, learning rate, depth, and L2 leaf regularization. Best configuration: `iterations=500, learning_rate=0.1, depth=4, l2_leaf_reg=9`.

Final tuned CatBoost:

| Metric | Value |
|---|---|
| R² | 0.7868 |
| Adjusted R² | 0.7844 |
| MSE | 0.05 |
| Cross-Validation R² | 75.61% |

### Linear Regression Assumption Checks
Run alongside the model comparison, not skipped just because Linear Regression wasn't the final choice:
- Linearity — predicted vs. actual scatter
- Independence of errors — ACF plot of residuals
- Homoscedasticity — residuals vs. predicted values
- Normality of errors — residual histogram
- Influential points — Cook's Distance against the 4/n threshold
- Multicollinearity — VIF on the final feature set, all values under 4 (`sqft_above` highest at 3.62) — no problematic multicollinearity among the retained features

### CatBoost Residual Diagnostics
Residual-vs-fitted plot (lowess-smoothed), residual histogram, and Q-Q plot to check the final model's error behavior, plus CatBoost's native feature importance.

## Results Summary

- Linear Regression explains 55.75% of price variance (R²=0.5575), with an 11-point gap between R² and cross-validated R² pointing to limited generalization from a purely linear fit.
- CatBoost (tuned) explains 78.68% of price variance (R²=0.7868), with MSE roughly half that of Linear Regression and a tighter CV gap (75.61%), indicating both better fit and better generalization.
- Multicollinearity is not a concern in the final feature set (all VIF < 4) — the R² gain from CatBoost comes from capturing non-linear relationships and feature interactions, not from linear-model instability.

## Tech Stack

- Data processing: pandas, NumPy
- Statistical modeling: statsmodels (OLS, AIC/BIC, VIF, Cook's Distance), SciPy
- ML models: scikit-learn (Linear Regression, Decision Tree, Gradient Boosting, Random Forest), XGBoost, CatBoost
- Visualization: matplotlib, seaborn

## Repository Structure

```
├── Regression-Based_Real_Estate_Valuation.ipynb   # Full pipeline: cleaning → EDA → feature engineering → modeling
├── data.csv                                         # Raw input (4,600 property records)
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn statsmodels xgboost catboost matplotlib seaborn scipy
jupyter notebook Regression-Based_Real_Estate_Valuation.ipynb
```

Run all cells sequentially — feature selection and modeling sections depend on the engineered features and log-transforms applied earlier in the notebook.

## Author

Shashank Shekhar — M.Sc. Statistics (Applied Statistics and Informatics), IIT Bombay
