"""
Tool 3: Dosage Calculator
=========================

A Python computation tool that calculates medicine dosage based on:
- Weight (kg)
- Age group (child/adult/elderly)
- Drug-specific dosing guidelines

This demonstrates a COMPUTATION tool in Agentic AI:
- No external API calls
- No database lookups
- Pure Python logic and math

Standard dosing formulas used:
- Paracetamol: 10-15 mg/kg per dose (max 4 doses/day)
- Ibuprofen: 5-10 mg/kg per dose (max 3 doses/day)
- Amoxicillin: 25-50 mg/kg/day in divided doses
"""

from langchain.tools import tool
from typing import Optional
from dataclasses import dataclass


@dataclass
class DosageGuideline:
    """Standard dosage guideline for a drug."""
    drug_name: str
    mg_per_kg_min: float  # Minimum mg per kg
    mg_per_kg_max: float  # Maximum mg per kg
    max_single_dose_mg: float  # Maximum single dose
    max_daily_dose_mg: float  # Maximum in 24 hours
    doses_per_day: int  # How many times per day
    min_hours_between: float  # Minimum hours between doses
    min_age_years: float  # Minimum age
    notes: str  # Additional notes


# Standard dosing guidelines
DOSAGE_GUIDELINES = {
    "paracetamol": DosageGuideline(
        drug_name="Paracetamol (Acetaminophen)",
        mg_per_kg_min=10.0,
        mg_per_kg_max=15.0,
        max_single_dose_mg=1000.0,  # 1g max single dose for adults
        max_daily_dose_mg=4000.0,   # 4g max daily for adults
        doses_per_day=4,
        min_hours_between=4.0,
        min_age_years=0.25,  # 3 months
        notes="Do not exceed 4 doses in 24 hours. Avoid alcohol."
    ),
    "acetaminophen": DosageGuideline(  # Alias for paracetamol
        drug_name="Acetaminophen (Paracetamol)",
        mg_per_kg_min=10.0,
        mg_per_kg_max=15.0,
        max_single_dose_mg=1000.0,
        max_daily_dose_mg=4000.0,
        doses_per_day=4,
        min_hours_between=4.0,
        min_age_years=0.25,
        notes="Do not exceed 4 doses in 24 hours. Avoid alcohol."
    ),
    "ibuprofen": DosageGuideline(
        drug_name="Ibuprofen",
        mg_per_kg_min=5.0,
        mg_per_kg_max=10.0,
        max_single_dose_mg=400.0,  # 400mg max single dose (OTC)
        max_daily_dose_mg=1200.0,  # 1200mg max daily (OTC)
        doses_per_day=3,
        min_hours_between=6.0,
        min_age_years=0.5,  # 6 months
        notes="Take with food. Not for children under 6 months."
    ),
    "amoxicillin": DosageGuideline(
        drug_name="Amoxicillin",
        mg_per_kg_min=25.0,
        mg_per_kg_max=50.0,
        max_single_dose_mg=500.0,
        max_daily_dose_mg=3000.0,
        doses_per_day=3,
        min_hours_between=8.0,
        min_age_years=0,
        notes="Complete full course. Take with or without food."
    ),
    "cetirizine": DosageGuideline(
        drug_name="Cetirizine",
        mg_per_kg_min=0.25,
        mg_per_kg_max=0.25,
        max_single_dose_mg=10.0,
        max_daily_dose_mg=10.0,
        doses_per_day=1,
        min_hours_between=24.0,
        min_age_years=2.0,
        notes="Once daily. May cause drowsiness."
    ),
}


def get_age_adjusted_limits(
    guideline: DosageGuideline, 
    age_group: str
) -> tuple[float, float]:
    """
    Adjust maximum doses based on age group.
    
    Returns (max_single_dose, max_daily_dose)
    """
    if age_group == "child":
        # Children: use lower limits
        max_single = min(guideline.max_single_dose_mg * 0.5, 500)  # Half adult max
        max_daily = min(guideline.max_daily_dose_mg * 0.5, 2000)
    elif age_group == "elderly":
        # Elderly: use lower limits
        max_single = guideline.max_single_dose_mg * 0.75
        max_daily = guideline.max_daily_dose_mg * 0.75
    else:  # adult
        max_single = guideline.max_single_dose_mg
        max_daily = guideline.max_daily_dose_mg
    
    return max_single, max_daily


def normalize_drug_name(drug: str) -> str:
    """Normalize drug name for lookup."""
    drug = drug.lower().strip()
    
    # Common aliases
    aliases = {
        "tylenol": "paracetamol",
        "crocin": "paracetamol",
        "dolo": "paracetamol",
        "calpol": "paracetamol",
        "advil": "ibuprofen",
        "motrin": "ibuprofen",
        "brufen": "ibuprofen",
        "augmentin": "amoxicillin",
        "zyrtec": "cetirizine",
    }
    
    return aliases.get(drug, drug)


@tool
def calculate_dosage(drug: str, weight_kg: float, age_group: str) -> str:
    """
    Calculate appropriate medication dosage based on weight and age.
    
    Use this tool when user asks about:
    - How much medicine to give/take
    - Dosage for a specific weight
    - Pediatric/child dosing
    - Dose calculation
    
    Input parameters:
    - drug: Name of the medication (generic or common brand names)
            Examples: "paracetamol", "ibuprofen", "Tylenol", "Advil"
    - weight_kg: Patient weight in kilograms
                 Examples: 20, 25.5, 70
    - age_group: One of "child", "adult", or "elderly"
    
    Output: Calculated dosage with safety information
    
    Supported drugs: paracetamol/acetaminophen, ibuprofen, amoxicillin, cetirizine
    (and common brand names like Tylenol, Advil, Dolo, etc.)
    """
    print(f"\n   🔧 Tool: calculate_dosage('{drug}', {weight_kg}, '{age_group}')")
    
    # Validate inputs
    if weight_kg <= 0:
        return "Error: Weight must be a positive number."
    
    if weight_kg > 200:
        return "Error: Weight seems unusually high. Please verify and try again."
    
    age_group = age_group.lower().strip()
    if age_group not in ["child", "adult", "elderly"]:
        return f"Error: age_group must be 'child', 'adult', or 'elderly'. Got: '{age_group}'"
    
    # Normalize drug name
    drug_normalized = normalize_drug_name(drug)
    
    # Look up guidelines
    if drug_normalized not in DOSAGE_GUIDELINES:
        supported = ", ".join(DOSAGE_GUIDELINES.keys())
        return (
            f"Dosage calculation not available for '{drug}'.\n\n"
            f"Supported medications: {supported}\n\n"
            f"For other medications, please consult a pharmacist or doctor, "
            f"or use the search_drug_database tool for general information."
        )
    
    guideline = DOSAGE_GUIDELINES[drug_normalized]
    
    # Check minimum age
    if age_group == "child":
        # Estimate age from weight (very rough)
        estimated_age_years = weight_kg / 3.0  # ~3kg per year for young children
        if estimated_age_years < guideline.min_age_years:
            return (
                f"⚠️ CAUTION: {guideline.drug_name} may not be suitable for very young children.\n\n"
                f"Minimum age: {guideline.min_age_years * 12:.0f} months\n"
                f"Estimated age based on weight: {estimated_age_years * 12:.0f} months\n\n"
                f"Please consult a pediatrician before administering."
            )
    
    # Calculate dosage
    min_dose = weight_kg * guideline.mg_per_kg_min
    max_dose = weight_kg * guideline.mg_per_kg_max
    recommended_dose = (min_dose + max_dose) / 2  # Middle of range
    
    # Apply age-adjusted limits
    max_single, max_daily = get_age_adjusted_limits(guideline, age_group)
    
    # Cap at maximum single dose
    if recommended_dose > max_single:
        recommended_dose = max_single
        min_dose = min(min_dose, max_single)
        max_dose = max_single
    
    # Calculate daily maximum
    doses_today_max = int(max_daily / recommended_dose)
    doses_today_max = min(doses_today_max, guideline.doses_per_day)
    
    # Build response
    lines = [
        f"💊 DOSAGE CALCULATION: {guideline.drug_name}",
        f"",
        f"Patient: {age_group.capitalize()}, {weight_kg} kg",
        f"",
        f"📋 RECOMMENDED DOSE:",
        f"   • Single dose: {recommended_dose:.0f} mg",
        f"   • Range: {min_dose:.0f} - {max_dose:.0f} mg per dose",
        f"   • Maximum doses per day: {doses_today_max}",
        f"   • Wait at least {guideline.min_hours_between:.0f} hours between doses",
        f"",
        f"📊 DAILY LIMITS:",
        f"   • Maximum single dose: {max_single:.0f} mg",
        f"   • Maximum daily total: {max_daily:.0f} mg",
        f"",
        f"ℹ️ Notes: {guideline.notes}",
        f"",
        f"─" * 40,
        f"⚠️ DISCLAIMER: This is general guidance only.",
        f"Always follow prescription instructions or consult a healthcare",
        f"professional for personalized medical advice."
    ]
    
    # Add syrup conversion for children
    if age_group == "child" and drug_normalized in ["paracetamol", "acetaminophen"]:
        # Common concentration: 120mg/5ml or 250mg/5ml
        syrup_120 = (recommended_dose / 120) * 5
        syrup_250 = (recommended_dose / 250) * 5
        
        lines.insert(-4, f"")
        lines.insert(-4, f"🍼 SYRUP CONVERSION (if using liquid):")
        lines.insert(-4, f"   • If 120mg/5ml concentration: {syrup_120:.1f} ml")
        lines.insert(-4, f"   • If 250mg/5ml concentration: {syrup_250:.1f} ml")
    
    result = "\n".join(lines)
    print(f"   📎 Calculated dose: {recommended_dose:.0f}mg")
    
    return result


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DOSAGE CALCULATOR TOOL TEST")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        ("paracetamol", 20.0, "child"),
        ("paracetamol", 70.0, "adult"),
        ("ibuprofen", 25.0, "child"),
        ("Tylenol", 15.0, "child"),  # Brand name test
        ("amoxicillin", 30.0, "child"),
    ]
    
    for drug, weight, age_group in test_cases:
        print(f"\n{'='*60}")
        print(f"Drug: {drug}, Weight: {weight}kg, Age: {age_group}")
        print("=" * 60)
        
        result = calculate_dosage.invoke({
            "drug": drug,
            "weight_kg": weight,
            "age_group": age_group
        })
        print(f"\n{result}")
