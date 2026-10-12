"""
Dataset generator for Real Estate Price Prediction and Dynamic Property Matchmaker.
Generates realistic property data with non-linear multi-attribute pricing dynamics.
"""

import os
import random
import numpy as np
import pandas as pd

LOCATIONS = [
    "Downtown",
    "Suburbs",
    "Waterfront",
    "Tech Corridor",
    "Historic District",
    "Hillside View",
    "Greenfield"
]

LOCATION_MULTIPLIERS = {
    "Downtown": 1.55,
    "Waterfront": 1.90,
    "Tech Corridor": 1.45,
    "Hillside View": 1.30,
    "Historic District": 1.15,
    "Suburbs": 1.00,
    "Greenfield": 0.82
}

PROPERTY_TYPES = ["Single Family", "Condo", "Townhouse", "Penthouse", "Villa"]

AMENITIES_POOL = [
    "Swimming Pool",
    "Fitness Center",
    "Solar Panels",
    "Private Garden",
    "Smart Home Automation",
    "24/7 Gated Security",
    "Scenic Balcony",
    "EV Fast Charger",
    "Home Cinema",
    "Wine Cellar"
]

AMENITY_VALUES = {
    "Swimming Pool": 28000,
    "Fitness Center": 15000,
    "Solar Panels": 18000,
    "Private Garden": 16000,
    "Smart Home Automation": 12000,
    "24/7 Gated Security": 14000,
    "Scenic Balcony": 9000,
    "EV Fast Charger": 8000,
    "Home Cinema": 22000,
    "Wine Cellar": 25000
}

ADJECTIVES = ["Luxury", "Contemporary", "Spacious", "Charming", "Modern", "Classic", "Serene", "Urban", "Exclusive", "Cozy"]
NOUNS = ["Haven", "Manor", "Estate", "Loft", "Retreat", "Residence", "Villa", "Oasis", "Penthouse", "Townhome"]

def generate_property_data(n_samples: int = 600, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    records = []
    
    for i in range(1, n_samples + 1):
        prop_id = f"PROP-{1000 + i}"
        location = random.choices(
            LOCATIONS,
            weights=[0.18, 0.28, 0.12, 0.16, 0.10, 0.08, 0.08],
            k=1
        )[0]
        
        prop_type = random.choices(
            PROPERTY_TYPES,
            weights=[0.40, 0.25, 0.20, 0.08, 0.07],
            k=1
        )[0]
        
        # Sqft tailored by property type
        if prop_type == "Condo":
            sqft = int(np.random.normal(950, 250))
            sqft = max(450, min(sqft, 2200))
            bedrooms = max(1, min(int(sqft / 450 + np.random.choice([0, 1])), 3))
            bathrooms = max(1.0, min(bedrooms + np.random.choice([-0.5, 0, 0.5]), 3.0))
            garage = np.random.choice([0, 1], p=[0.4, 0.6])
        elif prop_type == "Townhouse":
            sqft = int(np.random.normal(1650, 300))
            sqft = max(1100, min(sqft, 2800))
            bedrooms = max(2, min(int(sqft / 500 + np.random.choice([0, 1])), 4))
            bathrooms = max(1.5, min(bedrooms + np.random.choice([-0.5, 0, 0.5]), 3.5))
            garage = np.random.choice([1, 2], p=[0.6, 0.4])
        elif prop_type == "Penthouse":
            sqft = int(np.random.normal(2800, 500))
            sqft = max(1800, min(sqft, 4800))
            bedrooms = max(3, min(int(sqft / 600 + np.random.choice([0, 1])), 5))
            bathrooms = max(2.5, min(bedrooms + np.random.choice([0, 0.5, 1.0]), 5.0))
            garage = np.random.choice([2, 3], p=[0.5, 0.5])
        elif prop_type == "Villa":
            sqft = int(np.random.normal(3800, 700))
            sqft = max(2400, min(sqft, 6200))
            bedrooms = max(4, min(int(sqft / 650 + np.random.choice([0, 1])), 6))
            bathrooms = max(3.0, min(bedrooms + np.random.choice([0, 0.5, 1.0]), 6.0))
            garage = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        else: # Single Family
            sqft = int(np.random.normal(2200, 550))
            sqft = max(900, min(sqft, 4500))
            bedrooms = max(2, min(int(sqft / 550 + np.random.choice([0, 1])), 5))
            bathrooms = max(1.5, min(bedrooms + np.random.choice([-0.5, 0, 0.5]), 4.5))
            garage = np.random.choice([1, 2, 3], p=[0.25, 0.60, 0.15])
            
        age = int(max(0, min(np.random.exponential(12), 45)))
        
        # Distance to center depends partly on location
        if location == "Downtown":
            distance_to_center = round(np.random.uniform(0.3, 3.5), 1)
        elif location in ["Tech Corridor", "Historic District"]:
            distance_to_center = round(np.random.uniform(2.5, 9.0), 1)
        elif location in ["Waterfront", "Hillside View"]:
            distance_to_center = round(np.random.uniform(4.0, 15.0), 1)
        elif location == "Suburbs":
            distance_to_center = round(np.random.uniform(8.0, 22.0), 1)
        else: # Greenfield
            distance_to_center = round(np.random.uniform(15.0, 32.0), 1)
            
        # Amenities selection
        n_amenities = random.choices([1, 2, 3, 4, 5, 6], weights=[0.15, 0.25, 0.30, 0.18, 0.08, 0.04], k=1)[0]
        # Penthouses and Villas have higher probability of luxury amenities
        if prop_type in ["Penthouse", "Villa"]:
            n_amenities = min(len(AMENITIES_POOL), n_amenities + 2)
        prop_amenities = sorted(random.sample(AMENITIES_POOL, k=n_amenities))
        
        # Title
        title = f"{random.choice(ADJECTIVES)} {location} {random.choice(NOUNS)}"
        
        # Realistic Price Calculation:
        # Base value from sqft with non-linear scaling (e.g. sqft^1.05 for premium scale)
        sqft_val = (sqft ** 1.03) * 165
        
        # Room values
        rooms_val = (bedrooms * 24000) + (bathrooms * 19000) + (garage * 16000)
        
        # Amenities value sum
        amenity_val = sum(AMENITY_VALUES[a] for a in prop_amenities)
        
        # Age depreciation factor: newer is more valuable
        age_factor = max(0.68, 1.0 - (age * 0.009))
        
        # Distance factor: closer to center is more valuable
        distance_factor = max(0.72, 1.0 - (distance_to_center * 0.011))
        
        # Combined subtotal
        subtotal = (sqft_val + rooms_val + amenity_val) * age_factor * distance_factor
        
        # Location multiplier
        location_mult = LOCATION_MULTIPLIERS[location]
        
        # Base price calculation with stochastic noise (heteroscedastic: slightly higher variance for luxury)
        noise = np.random.normal(0, 18000 + (subtotal * 0.025))
        calculated_price = (subtotal * location_mult) + 85000 + noise
        
        # Round to nearest thousand
        price = int(round(calculated_price / 1000.0) * 1000)
        price = max(110000, price) # Floor to realistic minimum
        
        records.append({
            "property_id": prop_id,
            "title": title,
            "property_type": prop_type,
            "location": location,
            "sqft": sqft,
            "bedrooms": int(bedrooms),
            "bathrooms": float(bathrooms),
            "age": age,
            "garage": int(garage),
            "distance_to_center_km": distance_to_center,
            "amenities": "; ".join(prop_amenities),
            "amenity_count": len(prop_amenities),
            "price": price
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_property_data(n_samples=650, random_seed=42)
    output_path = os.path.join("data", "real_estate_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} properties to {output_path}")
    print("Price range: ${:,} to ${:,}".format(df['price'].min(), df['price'].max()))
    print(df.head(3))
