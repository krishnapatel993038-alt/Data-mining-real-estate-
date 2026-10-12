"""
Formal Constraint Satisfaction Problem (CSP) definitions for Real Estate Property Matching.
Defines Buyer Preferences, Property Domain, Hard Constraints, and Soft Constraints.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

@dataclass
class BuyerPreference:
    max_budget: float
    min_budget: float = 0.0
    preferred_locations: List[str] = field(default_factory=list)
    min_bedrooms: int = 1
    min_bathrooms: float = 1.0
    essential_amenities: List[str] = field(default_factory=list)  # Hard constraints
    desired_amenities: List[str] = field(default_factory=list)    # Soft constraints
    max_commute_km: Optional[float] = None                       # Hard constraint if specified
    max_age_years: Optional[int] = None                          # Soft/hard constraint
    allow_relaxation: bool = True                                # Enable soft constraint relaxation

@dataclass
class ConstraintResult:
    is_satisfied: bool
    constraint_name: str
    details: str
    penalty_score: float = 0.0  # 0 if satisfied, > 0 if relaxed or violated

class RealEstateCSP:
    """
    Formal representation of a Property Matching CSP.
    Variables: Property X (assigned to a candidate from the inventory domain).
    Domain: Set of properties D = {prop_1, prop_2, ... prop_N}.
    Constraints: C_budget, C_location, C_bedrooms, C_bathrooms, C_amenities, C_commute.
    """
    def __init__(self, buyer: BuyerPreference, property_catalog: List[Dict[str, Any]]):
        self.buyer = buyer
        self.property_catalog = property_catalog
        
    def evaluate_budget_constraint(self, prop: Dict[str, Any], budget_margin: float = 0.0) -> ConstraintResult:
        price = prop["price"]
        effective_max = self.buyer.max_budget * (1.0 + budget_margin)
        if price <= effective_max:
            if price <= self.buyer.max_budget:
                return ConstraintResult(True, "Budget Constraint", f"${price:,} <= Max Budget ${int(self.buyer.max_budget):,}")
            else:
                pct = round(((price - self.buyer.max_budget) / self.buyer.max_budget) * 100, 1)
                return ConstraintResult(True, "Budget Constraint (Relaxed)", f"${price:,} exceeds budget by {pct}% (within +{int(budget_margin*100)}% relaxation)", penalty_score=pct * 0.5)
        diff = price - self.buyer.max_budget
        return ConstraintResult(False, "Budget Constraint", f"${price:,} exceeds budget by ${int(diff):,}", penalty_score=50.0)

    def evaluate_location_constraint(self, prop: Dict[str, Any], allow_any_loc: bool = False) -> ConstraintResult:
        if not self.buyer.preferred_locations or allow_any_loc:
            return ConstraintResult(True, "Location Constraint", f"Location '{prop['location']}' accepted (open preferences)")
        if prop["location"] in self.buyer.preferred_locations:
            return ConstraintResult(True, "Location Constraint", f"Location '{prop['location']}' matches preferred locations")
        return ConstraintResult(False, "Location Constraint", f"Location '{prop['location']}' not in {self.buyer.preferred_locations}", penalty_score=40.0)

    def evaluate_bedrooms_constraint(self, prop: Dict[str, Any], min_bed_relaxed: int = 0) -> ConstraintResult:
        target_beds = max(1, self.buyer.min_bedrooms - min_bed_relaxed)
        actual = prop["bedrooms"]
        if actual >= self.buyer.min_bedrooms:
            return ConstraintResult(True, "Bedrooms Constraint", f"{actual} beds >= {self.buyer.min_bedrooms} required")
        elif actual >= target_beds:
            return ConstraintResult(True, "Bedrooms Constraint (Relaxed)", f"{actual} beds meets relaxed requirement of {target_beds}", penalty_score=20.0)
        return ConstraintResult(False, "Bedrooms Constraint", f"{actual} beds < {self.buyer.min_bedrooms} required", penalty_score=35.0)

    def evaluate_bathrooms_constraint(self, prop: Dict[str, Any]) -> ConstraintResult:
        actual = prop["bathrooms"]
        if actual >= self.buyer.min_bathrooms:
            return ConstraintResult(True, "Bathrooms Constraint", f"{actual} baths >= {self.buyer.min_bathrooms} required")
        return ConstraintResult(False, "Bathrooms Constraint", f"{actual} baths < {self.buyer.min_bathrooms} required", penalty_score=25.0)

    def evaluate_amenities_constraint(self, prop: Dict[str, Any], max_missing_essential: int = 0) -> ConstraintResult:
        prop_amenities_str = prop.get("amenities", "")
        prop_amenities = [a.strip() for a in prop_amenities_str.split(";") if a.strip()] if isinstance(prop_amenities_str, str) else prop.get("amenities", [])
        
        missing = [a for a in self.buyer.essential_amenities if a not in prop_amenities]
        if len(missing) == 0:
            return ConstraintResult(True, "Essential Amenities", f"All {len(self.buyer.essential_amenities)} essential amenities satisfied")
        elif len(missing) <= max_missing_essential:
            return ConstraintResult(True, "Essential Amenities (Relaxed)", f"Relaxed: missing {len(missing)} amenities ({', '.join(missing)})", penalty_score=len(missing) * 15.0)
        return ConstraintResult(False, "Essential Amenities", f"Missing essential amenities: {', '.join(missing)}", penalty_score=len(missing) * 20.0)

    def evaluate_commute_constraint(self, prop: Dict[str, Any]) -> ConstraintResult:
        if self.buyer.max_commute_km is None:
            return ConstraintResult(True, "Commute Constraint", "No commute limit specified")
        dist = prop["distance_to_center_km"]
        if dist <= self.buyer.max_commute_km:
            return ConstraintResult(True, "Commute Constraint", f"{dist} km <= {self.buyer.max_commute_km} km max commute")
        return ConstraintResult(False, "Commute Constraint", f"{dist} km exceeds {self.buyer.max_commute_km} km commute", penalty_score=15.0)
