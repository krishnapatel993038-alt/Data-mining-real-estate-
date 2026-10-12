"""
FastAPI Server for Real Estate Price Prediction & Dynamic Property Matchmaker.
Integrates Data Mining Regression Models with AI Constraint Satisfaction Problem (CSP) Solver.
"""

import os
import sys
from typing import List, Optional, Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.regression_models import RealEstateModelSuite, LOCATIONS, AMENITIES_POOL
from ai_csp.csp_problem import BuyerPreference
from ai_csp.csp_solver import DynamicPropertyMatchmaker

app = FastAPI(
    title="Real Estate Price Prediction & Dynamic Property Matchmaker",
    description="Data Mining (Linear & Non-Linear Regression) + AI (Constraint Satisfaction Problem)",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Global State Initialization
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "real_estate_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "trained_suite.joblib")

# Load Dataset
if os.path.exists(DATA_PATH):
    df_catalog = pd.read_csv(DATA_PATH)
    property_catalog = df_catalog.to_dict(orient="records")
else:
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# Load or Train Model Suite
if os.path.exists(MODEL_PATH):
    model_suite = RealEstateModelSuite.load(MODEL_PATH)
else:
    model_suite = RealEstateModelSuite()
    model_suite.train_all(DATA_PATH)
    model_suite.save(MODEL_PATH)

# Initialize AI CSP Matchmaker
csp_matchmaker = DynamicPropertyMatchmaker(property_catalog)

# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    sqft: float = Field(..., ge=200, le=10000, description="Square footage")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=1.0, le=10.0, description="Number of bathrooms")
    location: str = Field(..., description="Neighborhood location")
    age: int = Field(5, ge=0, le=100, description="Property age in years")
    garage: int = Field(1, ge=0, le=5, description="Garage capacity")
    distance_to_center_km: float = Field(5.0, ge=0.5, le=50.0, description="Distance to city center (km)")
    amenities: List[str] = Field(default_factory=list, description="List of amenities")
    selected_model: str = Field("Random Forest (Non-Linear)", description="Preferred regression model")

class CSPMatchRequest(BaseModel):
    max_budget: float = Field(..., ge=50000, description="Maximum budget in USD")
    preferred_locations: List[str] = Field(default_factory=list, description="Allowed locations/neighborhoods")
    min_bedrooms: int = Field(1, ge=1, le=8, description="Minimum bedrooms required")
    min_bathrooms: float = Field(1.0, ge=1.0, le=8.0, description="Minimum bathrooms required")
    essential_amenities: List[str] = Field(default_factory=list, description="Must-have amenities (Hard constraints)")
    desired_amenities: List[str] = Field(default_factory=list, description="Nice-to-have amenities (Soft constraints)")
    max_commute_km: Optional[float] = Field(None, ge=1.0, le=50.0, description="Maximum distance to center")
    allow_relaxation: bool = Field(True, description="Enable intelligent soft relaxation if over-constrained")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    """Returns available filter options and metadata."""
    return {
        "locations": LOCATIONS,
        "amenities": AMENITIES_POOL,
        "models": list(model_suite.models.keys()),
        "total_properties": len(property_catalog)
    }

@app.get("/api/models/summary")
def get_models_summary():
    """Returns comparative metrics and regression statistics."""
    summary = model_suite.get_summary()
    return {
        "metrics": summary["metrics"],
        "slr_stats": summary["slr_stats"],
        "feature_importances": summary["feature_importances"],
        "feature_names": summary["feature_names"],
        "sample_points": summary.get("test_sample_points", {})
    }

@app.post("/api/models/predict")
def predict_price(req: PredictionRequest):
    """Predicts property price across models with confidence intervals."""
    try:
        input_data = req.dict()
        result = model_suite.predict(input_data, model_name=req.selected_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/csp/match")
def match_properties(req: CSPMatchRequest):
    """
    Solves buyer preference Constraint Satisfaction Problem (CSP)
    with Forward Checking, Trace Pruning, and Intelligent Relaxation.
    """
    try:
        pref = BuyerPreference(
            max_budget=req.max_budget,
            preferred_locations=req.preferred_locations,
            min_bedrooms=req.min_bedrooms,
            min_bathrooms=req.min_bathrooms,
            essential_amenities=req.essential_amenities,
            desired_amenities=req.desired_amenities,
            max_commute_km=req.max_commute_km,
            allow_relaxation=req.allow_relaxation
        )
        match_result = csp_matchmaker.match(pref)
        return match_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/properties")
def get_properties(
    location: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_beds: Optional[int] = None,
    sort_by: str = Query("price", pattern="^(price|sqft|age|distance_to_center_km)$"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=50)
):
    """Browse catalog with filtering, sorting, and pagination."""
    filtered = list(property_catalog)

    if location and location != "All":
        filtered = [p for p in filtered if p["location"] == location]
    if min_price is not None:
        filtered = [p for p in filtered if p["price"] >= min_price]
    if max_price is not None:
        filtered = [p for p in filtered if p["price"] <= max_price]
    if min_beds is not None:
        filtered = [p for p in filtered if p["bedrooms"] >= min_beds]

    reverse = (sort_order == "desc")
    filtered.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)

    total_count = len(filtered)
    start_idx = (page - 1) * limit
    paginated_items = filtered[start_idx:start_idx + limit]

    return {
        "total": total_count,
        "page": page,
        "limit": limit,
        "total_pages": (total_count + limit - 1) // limit,
        "items": paginated_items
    }

@app.get("/api/dataset/stats")
def get_dataset_stats():
    """Returns analytical aggregate stats for the real estate market."""
    df = df_catalog
    location_stats = df.groupby("location").agg(
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        avg_sqft=("sqft", "mean"),
        count=("property_id", "count")
    ).round(2).to_dict(orient="index")

    # Scatter sample for UI scatter plot (Price vs Sqft)
    scatter_sample = df[["sqft", "price", "location", "bedrooms"]].sample(min(120, len(df)), random_state=42).to_dict(orient="records")

    return {
        "total_properties": len(df),
        "overall_avg_price": round(float(df["price"].mean()), 2),
        "overall_min_price": int(df["price"].min()),
        "overall_max_price": int(df["price"].max()),
        "location_stats": location_stats,
        "scatter_sample": scatter_sample
    }

# ---------------------------------------------------------------------------
# Static Files & UI Serving
# ---------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to Real Estate Price Prediction & Dynamic Property Matchmaker API. UI loading..."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
