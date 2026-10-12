"""
Model training script for the Real Estate Price Prediction suite.
Trains Simple Linear Regression, Multiple Linear Regression, Polynomial,
Ridge, and Random Forest models on data/real_estate_data.csv.
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.regression_models import RealEstateModelSuite

def run_training():
    print("=" * 60)
    print("TRAINING REAL ESTATE REGRESSION MODELS...")
    print("=" * 60)
    
    suite = RealEstateModelSuite()
    data_path = os.path.join("data", "real_estate_data.csv")
    
    summary = suite.train_all(data_path)
    
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(f"{'Model Name':<35} | {'R² Score':<9} | {'RMSE ($)':<12} | {'MAE ($)':<10} | {'MAPE (%)':<8}")
    print("-" * 82)
    for model_name, m in summary["metrics"].items():
        print(f"{model_name:<35} | {m['r2_score']:<9.4f} | ${m['rmse']:<11,.2f} | ${m['mae']:<9,.2f} | {m['mape']:<7.2f}%")
        
    print("\n--- SIMPLE LINEAR REGRESSION FORMULA ---")
    print(summary["slr_stats"]["equation"])
    
    print("\n--- TOP PREDICTIVE FEATURES (RANDOM FOREST) ---")
    rf_feats = summary["feature_importances"].get("Random Forest (Non-Linear)", {})
    for feat, imp in list(rf_feats.items())[:8]:
        print(f"  • {feat:<28}: {imp*100:.2f}% importance")
        
    model_save_path = os.path.join("models", "trained_suite.joblib")
    suite.save(model_save_path)
    print(f"\nTrained models successfully serialized to: {model_save_path}")
    return suite

if __name__ == "__main__":
    run_training()
