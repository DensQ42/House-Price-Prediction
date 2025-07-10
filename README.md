# House Prices: Advanced Regression Techniques

## Overview

This project solves the regression problem from the Kaggle competition ["House Prices: Advanced Regression Techniques."](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) The goal is to predict the sale price of houses in Ames, Iowa, using various numerical and categorical features.

## Dataset

The dataset is provided by Kaggle and consists of training and test sets with detailed information about each house, including:

- Lot dimensions
- House quality and condition
- Year built and remodeled
- Basement, garage, and porch features
- Neighborhood and location specifics

The target variable is `SalePrice`.

## Project Structure

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/
│   ├── best_hyperparams.json
│   └── best_scores.json
├── output/
│   └── my_submission.csv
├── scripts/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── eda_utils.py
│   └── model_utils.py
├── .gitignore
├── solution.ipynb
└── README.md
```

## Key Steps

1. **Data Loading**

   - Read training and test datasets.
   - Performed initial exploration using a custom EDA function.

2. **Missing Value Handling**

   - Imputed missing values based on feature type and domain knowledge.
   - Verified datasets to ensure no remaining missing values.

3. **Feature Engineering**

   - Identified and removed outliers using visual analysis.
   - Log-transformed the target variable to correct skewness.
   - Corrected skewed feature distributions using the Yeo-Johnson transformation.
   - Applied feature scaling with StandardScaler.
   - Encoded categorical features:
     - One-hot encoding for low-cardinality features.
     - Frequency encoding for high-cardinality features.

4. **Modeling**

   - Trained multiple regression models: Linear Regression, ElasticNet, Lasso, Ridge, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
   - Performed hyperparameter tuning using cross-validation and Optuna-based search.
   - Evaluated models using RMSE.
   - Selected best-performing models based on cross-validation results.

5. **Ensemble Learning**

   - Combined selected models using a stacking regressor with RidgeCV as the final estimator.
   - Trained the stacking model and generated predictions for the test set.

6. **Submission**

   - Applied the inverse logarithmic transformation to predicted values.
   - Prepared and saved the final submission file.

## Results

| Model            | RMSE     |
| ---------------- | -------- |
| GradientBoosting | 0.113829 |
| Lasso            | 0.114161 |
| ElasticNet       | 0.114198 |
| Ridge            | 0.114667 |
| XGBoost          | 0.116735 |
| LGBM             | 0.117122 |
| RandomForest     | 0.133752 |
| KNN              | 0.161826 |
| DecisionTree     | 0.182320 |
| LinearRegression | Failed   |

The best-performing models were Gradient Boosting, Lasso, ElasticNet, XGBoost, and LGBM. The final stacking ensemble achieved the lowest RMSE on the test data.

## Key Takeaways

- Logarithmic and Yeo-Johnson transformations significantly improved feature distributions and model performance.
- Frequency encoding effectively handled high-cardinality categorical features.
- Stacking multiple models provided a robust final solution.
- Simpler models like DecisionTree and KNN underperformed compared to advanced ensemble techniques.

## Next Steps

- Further tune stacking weights and explore blending techniques.
- Test more advanced feature selection strategies.
- Explore alternative cross-validation strategies like GroupKFold.
- Conduct detailed error analysis to improve model robustness.

## How to Run

1. Clone the repository and navigate to the project directory.
2. Ensure the dataset files are located in the `data` folder.
3. Run the Jupyter Notebook `solution.ipynb` step by step.
4. Final submission will be saved in the `output` directory.

## Requirements

- Python 3.12
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- optuna


## Author

Denis Kurovskii

---

This project demonstrates a full machine learning pipeline including EDA, feature engineering, hyperparameter tuning, model evaluation, and ensembling.
