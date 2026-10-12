"""
Regression Models for Real Estate Price Prediction.
Includes:
- Simple Linear Regression (SLR)
- Multiple Linear Regression (MLR)
- Polynomial Regression (Degree 2)
- Random Forest Regressor (Non-Linear Ensemble)
- Ridge Regression (L2 Regularized)
Along with comprehensive evaluation metrics (R2, RMSE, MAE, MAPE),
residuals calculation, and feature importance analysis.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data.dataset_generator import AMENITIES_POOL, LOCATIONS

FEATURE_COLUMNS_NUMERIC = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age",
    "garage",
    "distance_to_center_km",
    "amenity_count"
]

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Transforms the raw DataFrame into model-ready features with one-hot encoding.
    """
    df_clean = df.copy()
    
    # One-hot encode locations
    for loc in LOCATIONS:
        df_clean[f"loc_{loc}"] = (df_clean["location"] == loc).astype(int)
        
    # Amenity flags
    for amenity in AMENITIES_POOL:
        col_name = f"has_{amenity.lower().replace(' ', '_').replace('/', '_')}"
        df_clean[col_name] = df_clean["amenities"].apply(
            lambda am_str: 1 if amenity in str(am_str) else 0
        )
        
    feature_cols = [c for c in df_clean.columns if c not in [
        "property_id", "title", "property_type", "location", "amenities", "price"
    ]]
    
    X = df_clean[feature_cols]
    y = df_clean["price"]
    return X, y, feature_cols

class RealEstateModelSuite:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, Dict[str, float]] = {}
        self.test_data: Dict[str, Any] = {}
        self.slr_stats: Dict[str, Any] = {}
        
    def train_all(self, data_path: str = "data/real_estate_data.csv") -> Dict[str, Any]:
        df = pd.read_csv(data_path)
        X, y, feature_names = preprocess_dataframe(df)
        self.feature_names = feature_names
        
        # Train / Test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
        # -------------------------------------------------------------
        # 1. Simple Linear Regression (Price vs sqft only)
        # -------------------------------------------------------------
        slr = LinearRegression()
        slr.fit(X_train[["sqft"]], y_train)
        y_pred_slr = slr.predict(X_test[["sqft"]])
        
        self.models["Simple Linear Regression"] = slr
        self.slr_stats = {
            "slope": float(slr.coef_[0]),
            "intercept": float(slr.intercept_),
            "equation": f"Price = ${slr.intercept_:,.0f} + ${slr.coef_[0]:,.2f} × sqft"
        }
        
        # -------------------------------------------------------------
        # 2. Multiple Linear Regression
        # -------------------------------------------------------------
        mlr = LinearRegression()
        mlr.fit(X_train, y_train)
        y_pred_mlr = mlr.predict(X_test)
        self.models["Multiple Linear Regression"] = mlr
        
        # Feature weights for MLR
        mlr_weights = {
            feat: float(coef) for feat, coef in zip(feature_names, mlr.coef_)
        }
        self.feature_importances["Multiple Linear Regression"] = dict(
            sorted(mlr_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        # -------------------------------------------------------------
        # 3. Polynomial Regression (Degree 2 on top numeric features)
        # -------------------------------------------------------------
        # We use a pipeline: PolyFeatures(degree=2, include_bias=False) + Ridge/LinearRegression
        poly_pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0))
        ])
        poly_pipeline.fit(X_train, y_train)
        y_pred_poly = poly_pipeline.predict(X_test)
        self.models["Polynomial Regression (Degree 2)"] = poly_pipeline
        
        # -------------------------------------------------------------
        # 4. Random Forest Regressor (Non-Linear Ensemble)
        # -------------------------------------------------------------
        rf = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        self.models["Random Forest (Non-Linear)"] = rf
        
        # Feature importances for RF
        rf_weights = {
            feat: float(imp) for feat, imp in zip(feature_names, rf.feature_importances_)
        }
        self.feature_importances["Random Forest (Non-Linear)"] = dict(
            sorted(rf_weights.items(), key=lambda x: x[1], reverse=True)
        )
        
        # -------------------------------------------------------------
        # 5. Ridge Regression (L2 Regularized Linear)
        # -------------------------------------------------------------
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        self.models["Ridge Regression"] = ridge
        
        # Calculate evaluation metrics
        preds_dict = {
            "Simple Linear Regression": y_pred_slr,
            "Multiple Linear Regression": y_pred_mlr,
            "Polynomial Regression (Degree 2)": y_pred_poly,
            "Random Forest (Non-Linear)": y_pred_rf,
            "Ridge Regression": y_pred_ridge
        }
        
        self.metrics = {}
        residuals_sample = {}
        
        for name, pred in preds_dict.items():
            r2 = float(r2_score(y_test, pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))
            mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100)
            
            self.metrics[name] = {
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "mape": round(mape, 2)
            }
            
            # Sample first 40 test points for scatter visualization
            residuals_sample[name] = [
                {
                    "actual": float(act),
                    "predicted": float(pr),
                    "residual": float(act - pr)
                }
                for act, pr in zip(y_test.values[:40], pred[:40])
            ]
            
        self.test_data = {
            "y_test_sample": y_test.values[:40].tolist(),
            "residuals_sample": residuals_sample,
            "sqft_test_sample": X_test["sqft"].values[:40].tolist()
        }
        
        return self.get_summary()
        
    def get_summary(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "slr_stats": self.slr_stats,
            "test_sample_points": self.test_data
        }
        
    def predict(self, input_data: Dict[str, Any], model_name: str = "Random Forest (Non-Linear)") -> Dict[str, Any]:
        """
        Predicts price for a property given its attributes.
        """
        if not self.models:
            raise ValueError("Models are not trained yet.")
            
        sqft = float(input_data.get("sqft", 1500))
        bedrooms = int(input_data.get("bedrooms", 3))
        bathrooms = float(input_data.get("bathrooms", 2.0))
        age = int(input_data.get("age", 5))
        garage = int(input_data.get("garage", 1))
        distance_to_center_km = float(input_data.get("distance_to_center_km", 5.0))
        location = input_data.get("location", "Downtown")
        amenities_list = input_data.get("amenities", [])
        if isinstance(amenities_list, str):
            amenities_list = [a.strip() for a in amenities_list.split(";") if a.strip()]
            
        amenity_count = len(amenities_list)
        
        # Build feature vector
        vector_dict = {
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age": age,
            "garage": garage,
            "distance_to_center_km": distance_to_center_km,
            "amenity_count": amenity_count
        }
        
        for loc in LOCATIONS:
            vector_dict[f"loc_{loc}"] = 1 if loc == location else 0
            
        for am in AMENITIES_POOL:
            col_name = f"has_{am.lower().replace(' ', '_').replace('/', '_')}"
            vector_dict[col_name] = 1 if am in amenities_list else 0
            
        # Ensure all columns exist in expected order
        full_row = [vector_dict.get(c, 0) for c in self.feature_names]
        X_df = pd.DataFrame([full_row], columns=self.feature_names)
        
        predictions_all = {}
        for m_name, model in self.models.items():
            if m_name == "Simple Linear Regression":
                pred = model.predict(pd.DataFrame([[sqft]], columns=["sqft"]))[0]
            else:
                pred = model.predict(X_df)[0]
            predictions_all[m_name] = round(float(pred), 2)
            
        selected_pred = predictions_all.get(model_name, predictions_all["Random Forest (Non-Linear)"])
        
        # Price per sqft estimate
        price_per_sqft = round(selected_pred / max(sqft, 1), 2)
        
        # Estimated confidence range (using RMSE of selected model)
        rmse_val = self.metrics.get(model_name, {}).get("rmse", 25000)
        lower_bound = max(50000, round(selected_pred - (1.28 * rmse_val), 0))
        upper_bound = round(selected_pred + (1.28 * rmse_val), 0)
        
        return {
            "selected_model": model_name,
            "estimated_price": int(round(selected_pred)),
            "price_per_sqft": price_per_sqft,
            "confidence_interval_80": {
                "lower": int(lower_bound),
                "upper": int(upper_bound)
            },
            "all_model_predictions": predictions_all,
            "input_summary": {
                "sqft": sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "location": location,
                "amenities": amenities_list,
                "age": age,
                "garage": garage,
                "distance_km": distance_to_center_km
            }
        }

    def save(self, filepath: str = "models/trained_suite.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
    @staticmethod
    def load(filepath: str = "models/trained_suite.joblib") -> "RealEstateModelSuite":
        return joblib.load(filepath)
