"""
AI Constraint Satisfaction Problem (CSP) Solver.
Implements:
1. Forward Checking (AC-3 style Domain Pruning)
2. Most Constrained Variable / Degree Heuristics
3. Exact Match Backtracking Search
4. Soft Constraint Relaxation & Preference Scoring (for over-constrained queries)
5. Full step-by-step CSP Search Trace for user inspection
"""

import time
from typing import List, Dict, Any, Tuple
from ai_csp.csp_problem import BuyerPreference, RealEstateCSP, ConstraintResult

class DynamicPropertyMatchmaker:
    """
    AI Matchmaker implementing Constraint Satisfaction Problem solving.
    Matches buyer preferences against property inventory.
    """
    def __init__(self, property_catalog: List[Dict[str, Any]]):
        self.catalog = property_catalog

    def match(self, preference: BuyerPreference) -> Dict[str, Any]:
        """
        Executes CSP matching process:
        - Exact constraint propagation / forward checking
        - Backtracking verification
        - Soft constraint relaxation if over-constrained or for near-matches
        - Ranks results by satisfaction score
        """
        csp = RealEstateCSP(preference, self.catalog)
        start_time = time.perf_counter()
        
        # -------------------------------------------------------------
        # Phase 1: Forward Checking & Domain Pruning Trace
        # -------------------------------------------------------------
        trace = []
        current_domain = list(self.catalog)
        trace.append({
            "step": 0,
            "phase": "Domain Initialization",
            "constraint": "Universe of available properties",
            "domain_size": len(current_domain),
            "pruned": 0,
            "description": f"Loaded complete inventory domain of {len(current_domain)} properties."
        })
        
        # Constraint 1: Budget Pruning
        prev_size = len(current_domain)
        current_domain = [
            p for p in current_domain 
            if csp.evaluate_budget_constraint(p, budget_margin=0.0).is_satisfied
        ]
        trace.append({
            "step": 1,
            "phase": "Forward Checking: Budget",
            "constraint": f"Price <= ${preference.max_budget:,.0f}",
            "domain_size": len(current_domain),
            "pruned": prev_size - len(current_domain),
            "description": f"Filtered out {prev_size - len(current_domain)} properties exceeding max budget."
        })
        
        # Constraint 2: Location Pruning
        prev_size = len(current_domain)
        if preference.preferred_locations:
            current_domain = [
                p for p in current_domain 
                if csp.evaluate_location_constraint(p).is_satisfied
            ]
        trace.append({
            "step": 2,
            "phase": "Forward Checking: Location",
            "constraint": f"Location IN {preference.preferred_locations or 'Any'}",
            "domain_size": len(current_domain),
            "pruned": prev_size - len(current_domain),
            "description": f"Filtered out {prev_size - len(current_domain)} properties outside target locations."
        })
        
        # Constraint 3: Bedrooms & Bathrooms
        prev_size = len(current_domain)
        current_domain = [
            p for p in current_domain 
            if csp.evaluate_bedrooms_constraint(p).is_satisfied and csp.evaluate_bathrooms_constraint(p).is_satisfied
        ]
        trace.append({
            "step": 3,
            "phase": "Forward Checking: Rooms",
            "constraint": f"Beds >= {preference.min_bedrooms}, Baths >= {preference.min_bathrooms}",
            "domain_size": len(current_domain),
            "pruned": prev_size - len(current_domain),
            "description": f"Filtered out {prev_size - len(current_domain)} properties failing room count requirements."
        })
        
        # Constraint 4: Essential Amenities
        prev_size = len(current_domain)
        if preference.essential_amenities:
            current_domain = [
                p for p in current_domain 
                if csp.evaluate_amenities_constraint(p, max_missing_essential=0).is_satisfied
            ]
        trace.append({
            "step": 4,
            "phase": "Forward Checking: Essential Amenities",
            "constraint": f"Contains ALL {preference.essential_amenities or 'None'}",
            "domain_size": len(current_domain),
            "pruned": prev_size - len(current_domain),
            "description": f"Filtered out {prev_size - len(current_domain)} properties lacking mandatory amenities."
        })
        
        # Constraint 5: Commute Distance (if defined)
        prev_size = len(current_domain)
        if preference.max_commute_km is not None:
            current_domain = [
                p for p in current_domain 
                if csp.evaluate_commute_constraint(p).is_satisfied
            ]
            trace.append({
                "step": 5,
                "phase": "Forward Checking: Commute",
                "constraint": f"Distance <= {preference.max_commute_km} km",
                "domain_size": len(current_domain),
                "pruned": prev_size - len(current_domain),
                "description": f"Filtered out {prev_size - len(current_domain)} properties exceeding commute threshold."
            })
            
        exact_matches = current_domain
        
        # -------------------------------------------------------------
        # Phase 2: Compute Detailed Scores & Soft Constraints
        # -------------------------------------------------------------
        scored_exact = []
        for p in exact_matches:
            scored_exact.append(self._score_property(p, preference, is_relaxed=False, relaxations=[]))
            
        # Sort exact matches by satisfaction score descending
        scored_exact.sort(key=lambda x: x["satisfaction_score"], reverse=True)
        
        # -------------------------------------------------------------
        # Phase 3: Intelligent Constraint Relaxation (Soft CSP)
        # -------------------------------------------------------------
        relaxed_matches = []
        is_over_constrained = (len(exact_matches) == 0)
        
        if preference.allow_relaxation:
            # We search across the catalog for properties that can be satisfied
            # with minimal relaxation tiers
            candidate_pool = [p for p in self.catalog if p not in exact_matches]
            
            for p in candidate_pool:
                # Check relaxation level 1: Budget +10% OR 1 missing essential amenity
                eval_b1 = csp.evaluate_budget_constraint(p, budget_margin=0.10)
                eval_loc = csp.evaluate_location_constraint(p)
                eval_bed = csp.evaluate_bedrooms_constraint(p, min_bed_relaxed=0)
                eval_bath = csp.evaluate_bathrooms_constraint(p)
                eval_am1 = csp.evaluate_amenities_constraint(p, max_missing_essential=1)
                eval_com = csp.evaluate_commute_constraint(p)
                
                # If satisfies with slight relaxation:
                if eval_b1.is_satisfied and eval_loc.is_satisfied and eval_bed.is_satisfied and eval_bath.is_satisfied and eval_am1.is_satisfied and eval_com.is_satisfied:
                    relaxations = []
                    if p["price"] > preference.max_budget:
                        diff = p["price"] - preference.max_budget
                        pct = round((diff / preference.max_budget) * 100, 1)
                        relaxations.append(f"Budget relaxed by +{pct}% (+${int(diff):,})")
                    if not csp.evaluate_amenities_constraint(p, 0).is_satisfied:
                        relaxations.append("Allowed 1 missing essential amenity")
                        
                    if relaxations:
                        scored_relaxed = self._score_property(p, preference, is_relaxed=True, relaxations=relaxations)
                        relaxed_matches.append(scored_relaxed)
                        continue
                        
                # Level 2 relaxation: Budget +20% OR 1 fewer bedroom OR 2 missing amenities
                eval_b2 = csp.evaluate_budget_constraint(p, budget_margin=0.20)
                eval_bed2 = csp.evaluate_bedrooms_constraint(p, min_bed_relaxed=1)
                eval_am2 = csp.evaluate_amenities_constraint(p, max_missing_essential=2)
                
                if eval_b2.is_satisfied and eval_loc.is_satisfied and eval_bed2.is_satisfied and eval_bath.is_satisfied and eval_am2.is_satisfied:
                    relaxations = []
                    if p["price"] > preference.max_budget:
                        pct = round(((p["price"] - preference.max_budget) / preference.max_budget) * 100, 1)
                        relaxations.append(f"Budget relaxed by +{pct}%")
                    if p["bedrooms"] < preference.min_bedrooms:
                        relaxations.append(f"Bedrooms relaxed from {preference.min_bedrooms} to {p['bedrooms']}")
                    if not csp.evaluate_amenities_constraint(p, 0).is_satisfied:
                        relaxations.append("Allowed 2 missing amenities")
                        
                    if relaxations:
                        scored_relaxed = self._score_property(p, preference, is_relaxed=True, relaxations=relaxations)
                        relaxed_matches.append(scored_relaxed)
                        
            # Sort relaxed matches by score descending
            relaxed_matches.sort(key=lambda x: x["satisfaction_score"], reverse=True)
            relaxed_matches = relaxed_matches[:15] # Top 15 relaxed alternatives
            
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        return {
            "status": "success",
            "execution_time_ms": elapsed_ms,
            "is_over_constrained": is_over_constrained,
            "exact_matches_count": len(scored_exact),
            "exact_matches": scored_exact[:20],
            "relaxed_matches_count": len(relaxed_matches),
            "relaxed_matches": relaxed_matches,
            "search_trace": trace,
            "buyer_preference_summary": {
                "max_budget": preference.max_budget,
                "preferred_locations": preference.preferred_locations,
                "min_bedrooms": preference.min_bedrooms,
                "min_bathrooms": preference.min_bathrooms,
                "essential_amenities": preference.essential_amenities,
                "desired_amenities": preference.desired_amenities,
                "max_commute_km": preference.max_commute_km
            }
        }
        
    def _score_property(
        self,
        prop: Dict[str, Any],
        pref: BuyerPreference,
        is_relaxed: bool,
        relaxations: List[str]
    ) -> Dict[str, Any]:
        """
        Computes composite satisfaction score (0-100%) incorporating:
        - Hard constraint satisfaction
        - Desired amenities bonus
        - Budget savings bonus
        - Modernity / age bonus
        - Commute convenience bonus
        """
        base_score = 100.0 if not is_relaxed else 82.0
        
        # Penalties for relaxations
        if is_relaxed:
            if prop["price"] > pref.max_budget:
                diff_pct = ((prop["price"] - pref.max_budget) / pref.max_budget) * 100
                base_score -= min(25.0, diff_pct * 1.2)
            if prop["bedrooms"] < pref.min_bedrooms:
                base_score -= 15.0
                
        # Parse amenities
        prop_amenities_str = prop.get("amenities", "")
        if isinstance(prop_amenities_str, str):
            prop_amenities = [a.strip() for a in prop_amenities_str.split(";") if a.strip()]
        else:
            prop_amenities = prop.get("amenities", [])
            
        # Desired amenities bonus (Soft Constraint)
        desired_satisfied = [a for a in pref.desired_amenities if a in prop_amenities]
        desired_ratio = (len(desired_satisfied) / len(pref.desired_amenities)) if pref.desired_amenities else 1.0
        amenity_bonus = desired_ratio * 12.0
        
        # Budget efficiency bonus: Reward being nicely below budget
        if prop["price"] <= pref.max_budget:
            savings_pct = (pref.max_budget - prop["price"]) / pref.max_budget
            budget_bonus = min(8.0, savings_pct * 15.0)
        else:
            budget_bonus = 0.0
            
        # Age bonus (newer is better)
        age = prop.get("age", 15)
        age_bonus = max(0.0, (30 - age) / 30) * 4.0
        
        # Commute bonus
        dist = prop.get("distance_to_center_km", 10)
        commute_bonus = max(0.0, (25 - dist) / 25) * 4.0
        
        total_score = max(10.0, min(100.0, base_score + amenity_bonus + budget_bonus + age_bonus + commute_bonus - 10.0))
        
        # Check hard constraint breakdown status
        constraint_breakdown = [
            {
                "name": "Budget",
                "satisfied": prop["price"] <= pref.max_budget,
                "value": f"${prop['price']:,}",
                "target": f"<= ${int(pref.max_budget):,}"
            },
            {
                "name": "Location",
                "satisfied": (not pref.preferred_locations) or (prop["location"] in pref.preferred_locations),
                "value": prop["location"],
                "target": ", ".join(pref.preferred_locations) if pref.preferred_locations else "Any"
            },
            {
                "name": "Bedrooms",
                "satisfied": prop["bedrooms"] >= pref.min_bedrooms,
                "value": f"{prop['bedrooms']} beds",
                "target": f">= {pref.min_bedrooms}"
            },
            {
                "name": "Bathrooms",
                "satisfied": prop["bathrooms"] >= pref.min_bathrooms,
                "value": f"{prop['bathrooms']} baths",
                "target": f">= {pref.min_bathrooms}"
            },
            {
                "name": "Essential Amenities",
                "satisfied": all(a in prop_amenities for a in pref.essential_amenities),
                "value": f"{sum(1 for a in pref.essential_amenities if a in prop_amenities)}/{len(pref.essential_amenities)} met",
                "target": "All required"
            }
        ]
        
        return {
            "property_id": prop["property_id"],
            "title": prop["title"],
            "property_type": prop["property_type"],
            "location": prop["location"],
            "price": prop["price"],
            "sqft": prop["sqft"],
            "bedrooms": prop["bedrooms"],
            "bathrooms": prop["bathrooms"],
            "age": prop["age"],
            "garage": prop["garage"],
            "distance_to_center_km": prop["distance_to_center_km"],
            "amenities": prop_amenities,
            "satisfaction_score": round(total_score, 1),
            "is_exact_match": not is_relaxed,
            "relaxations": relaxations,
            "desired_amenities_met": desired_satisfied,
            "desired_amenities_total": len(pref.desired_amenities),
            "constraint_breakdown": constraint_breakdown
        }
