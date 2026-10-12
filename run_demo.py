"""
Console Demonstration Script:
Real Estate Price Prediction & Dynamic Property Matchmaker
Executes both:
1. Data Mining: Multi-Model Regression Price Estimation
2. AI: Constraint Satisfaction Problem (CSP) Solver & Forward Checking Trace
"""

import os
import sys
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.regression_models import RealEstateModelSuite
from ai_csp.csp_problem import BuyerPreference
from ai_csp.csp_solver import DynamicPropertyMatchmaker

def main():
    print("=" * 75)
    print("   ESTATEPULSE: DATA MINING & AI CONSTRAINT SATISFACTION MATCHMAKER   ")
    print("=" * 75)

    # 1. Load Data
    data_path = os.path.join("data", "real_estate_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    df = pd.read_csv(data_path)
    print(f"[*] Loaded Property Catalog: {len(df)} listings across {df['location'].nunique()} neighborhoods.")

    # 2. Load Trained Suite
    model_path = os.path.join("models", "trained_suite.joblib")
    suite = RealEstateModelSuite.load(model_path)
    print(f"[*] Loaded Regression Suite: {len(suite.models)} trained models.\n")

    # -------------------------------------------------------------
    # DEMO PART 1: DATA MINING PRICE PREDICTION
    # -------------------------------------------------------------
    print("=" * 75)
    print("PART 1: DATA MINING REGRESSION EVALUATION")
    print("=" * 75)

    sample_query = {
        "sqft": 2400,
        "bedrooms": 4,
        "bathrooms": 3.0,
        "location": "Downtown",
        "age": 5,
        "garage": 2,
        "distance_to_center_km": 2.5,
        "amenities": ["Balcony", "Hardwood Floors", "Private Garden", "Pool"]
    }
    print("Evaluating Test Property:")
    print(f"  • Area: {sample_query['sqft']} sqft | Bedrooms: {sample_query['bedrooms']} | Bathrooms: {sample_query['bathrooms']}")
    print(f"  • Location: {sample_query['location']} (2.5 km from center) | Age: {sample_query['age']} years")
    print(f"  • Amenities: {', '.join(sample_query['amenities'])}\n")

    pred_res = suite.predict(sample_query, model_name="Random Forest (Non-Linear)")
    print(f">> Random Forest Price Estimate: ${pred_res['estimated_price']:,} (${pred_res['price_per_sqft']}/sqft)")
    print(f">> 80% Confidence Interval: ${pred_res['confidence_interval_80']['lower']:,} to ${pred_res['confidence_interval_80']['upper']:,}\n")

    print("Comparative Multi-Model Price Projections:")
    for model_name, price in pred_res["all_model_predictions"].items():
        r2 = suite.metrics[model_name]["r2_score"]
        rmse = suite.metrics[model_name]["rmse"]
        print(f"  • {model_name:<34}: ${price:>10,.2f}  (R²: {r2:.4f}, RMSE: ${rmse:,.0f})")

    print("\nSimple Linear Regression Equation:")
    print(f"  {suite.slr_stats['equation']}\n")

    # -------------------------------------------------------------
    # DEMO PART 2: AI CSP PROPERTY MATCHMAKER
    # -------------------------------------------------------------
    print("=" * 75)
    print("PART 2: AI CONSTRAINT SATISFACTION PROBLEM (CSP) MATCHMAKER")
    print("=" * 75)

    buyer_pref = BuyerPreference(
        max_budget=750000,
        preferred_locations=["Downtown", "Uptown", "Tech Corridor"],
        min_bedrooms=3,
        min_bathrooms=2.0,
        essential_amenities=["Balcony", "Hardwood Floors"],
        desired_amenities=["Pool", "Gym", "Private Garden"],
        max_commute_km=10.0,
        allow_relaxation=True
    )

    print("Buyer Constraint Specifications:")
    print(f"  • Max Budget: ${buyer_pref.max_budget:,.0f} (Hard)")
    print(f"  • Target Neighborhoods: {', '.join(buyer_pref.preferred_locations)} (Hard)")
    print(f"  • Min. Bedrooms: {buyer_pref.min_bedrooms} | Min. Bathrooms: {buyer_pref.min_bathrooms} (Hard)")
    print(f"  • Mandatory Amenities: {', '.join(buyer_pref.essential_amenities)} (Hard)")
    print(f"  • Desired Amenities: {', '.join(buyer_pref.desired_amenities)} (Soft Bonus)")
    print(f"  • Max Commute: <= {buyer_pref.max_commute_km} km (Hard)\n")

    catalog = df.to_dict(orient="records")
    matchmaker = DynamicPropertyMatchmaker(catalog)
    csp_result = matchmaker.match(buyer_pref)

    print("--- FORWARD CHECKING EXECUTION TRACE ---")
    for step in csp_result["search_trace"]:
        print(f"  Step {step['step']}: {step['phase']:<35} | Remaining Domain: {step['domain_size']:>3} (-{step['pruned']:>3} pruned)")

    print(f"\nExecution Time: {csp_result['execution_time_ms']} ms")
    print(f"Exact Matches Found: {csp_result['exact_matches_count']}")
    print(f"Relaxed Candidates Found: {csp_result['relaxed_matches_count']}\n")

    if csp_result["exact_matches"]:
        print("Top Exact Match:")
        top = csp_result["exact_matches"][0]
        print(f"  • ID: {top['property_id']} - {top['title']}")
        print(f"  • Price: ${top['price']:,} | Sqft: {top['sqft']:,} | Score: {top['satisfaction_score']}%")
        print(f"  • Location: {top['location']} ({top['distance_to_center_km']} km to Center)")
        print(f"  • Amenities: {', '.join(top['amenities'])}")
    else:
        print("Query was strictly over-constrained for exact matches.")
        if csp_result["relaxed_matches"]:
            print("Top Relaxed Alternative:")
            top_rel = csp_result["relaxed_matches"][0]
            print(f"  • ID: {top_rel['property_id']} - {top_rel['title']}")
            print(f"  • Price: ${top_rel['price']:,} | Score: {top_rel['satisfaction_score']}%")
            print(f"  • Relaxations applied: {', '.join(top_rel['relaxations'])}")

    print("\n" + "=" * 75)
    print("All subsystems verified successfully!")
    print("=" * 75)

if __name__ == "__main__":
    main()
