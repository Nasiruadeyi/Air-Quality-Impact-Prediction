import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    models = {}

    # -----------------------
    # 1. Baseline: Linear Regression
    # -----------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results["Linear Regression"] = evaluate_model(lr, X_test, y_test)
    models["Linear Regression"] = lr

    # -----------------------
    # 2. XGBoost
    # -----------------------
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    xgb.fit(X_train, y_train)
    results["XGBoost"] = evaluate_model(xgb, X_test, y_test)
    models["XGBoost"] = xgb

    # -----------------------
    # 3. LightGBM
    # -----------------------
    lgbm = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    lgbm.fit(X_train, y_train)
    results["LightGBM"] = evaluate_model(lgbm, X_test, y_test)
    models["LightGBM"] = lgbm

    # -----------------------
    # Pick best model (highest R²)
    # -----------------------
    best_model_name = max(results, key=lambda m: results[m]["R2"])
    best_model = models[best_model_name]

    # Ensure models/ exists
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(best_model, f"models/best_model.pkl")

    return results, best_model_name

