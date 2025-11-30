# 🌬️ Air Quality Impact Prediction

## 📌 Project Overview
End-to-end ML project analyzing air quality’s impact on human health. Includes **data cleaning, EDA, feature engineering, regression modeling, hyperparameter tuning** (XGBoost & LightGBM), and evaluation. Predicts `HealthImpactScore` with metrics (MAE, RMSE, R²) and feature importance insights.

## 🗂️ Dataset
- CSV: `air_quality_health_impact_data.csv`
- Features include environmental and operational factors affecting health impact.
- Target: `HealthImpactScore`

## 🔍 Exploratory Data Analysis
- Distribution of target values  
- Correlation between pollutants and health impact  
- Key visualizations in `/images`

## 🤖 Machine Learning Models
- Baseline regression: Linear, Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees, GradientBoosting, KNN, SVR, XGBoost, LightGBM  
- Hyperparameter tuning: XGBoost & LightGBM  
- Evaluation metrics: MAE, RMSE, R²  

## 📈 Feature Importance
- LightGBM feature importance chart in `/images/feature_importance.png`

## 🛠️ How to Run
1. Clone the repo
2. Install requirements:  
   ```bash
   python -m pip install -r requirements.txt
3. Open notebooks/modeling.ipynb to run EDA and modeling

## 📂 Folder Structure
```bash
Air-Quality-Impact-Prediction/
│
├── air_quality_health_impact_data.csv   # dataset
│
├── notebooks/
│   └── modeling.ipynb                       # all analysis & modeling code
│
├── src/
│   ├── preprocessing.py                     # optional modular functions
│   └── train_model.py                       # optional modular training code
│
├── images/                                  # store plots & visualizations
│   └── feature_importance.png
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
