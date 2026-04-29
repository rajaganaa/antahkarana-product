"""
Tool 2: FDA Adverse Events API
==============================

This tool calls the OpenFDA API to get REAL adverse event reports.

API Documentation: https://open.fda.gov/apis/drug/event/

Features:
- Free, no API key required
- Real-world reported side effects
- Frequency data (how often reported)
- Serious reaction information
"""

import requests
from langchain.tools import tool
from typing import Optional


# OpenFDA API endpoint
FDA_API_BASE = "https://api.fda.gov/drug/event.json"


def query_fda_api(drug_name: str, limit: int = 5) -> Optional[dict]:
    """
    Query the OpenFDA adverse events API.
    """
    try:
        # Simpler, more specific query
        params = {
            "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
            "limit": limit
        }
        
        print(f"   [FDA API] Querying for: {drug_name}")
        response = requests.get(FDA_API_BASE, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"error": "not_found", "message": "No adverse events found"}
        else:
            print(f"   [FDA API] Error status: {response.status_code}")
            return {"error": "api_error", "status": response.status_code}
            
    except requests.Timeout:
        return {"error": "timeout", "message": "API request timed out"}
    except requests.RequestException as e:
        return {"error": "request_failed", "message": str(e)}


def get_reaction_counts(drug_name: str, top_n: int = 10) -> Optional[dict]:
    """
    Get counts of most common adverse reactions for a specific drug.
    """
    try:
        # More specific query - search by generic name
        params = {
            "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": top_n
        }
        
        print(f"   [FDA API] Getting reaction counts for: {drug_name}")
        response = requests.get(FDA_API_BASE, params=params, timeout=10)
        print(f"   [FDA API] Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   [FDA API] Got {len(data.get('results', []))} reaction types")
            return data
        else:
            # Try alternative search with brand name field
            params["search"] = f'patient.drug.medicinalproduct:"{drug_name}"'
            response = requests.get(FDA_API_BASE, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            return None
            
    except Exception as e:
        print(f"   [FDA API] Error: {e}")
        return None


@tool
def get_fda_adverse_events(drug_name: str) -> str:
    """
    Get real-world adverse event reports from the FDA database.
    
    Use this tool when you need:
    - Real reported side effects from patients/doctors
    - How frequently certain reactions are reported
    - Serious adverse event information
    - Real-world safety data (not just clinical trials)
    
    Input: Drug name (generic name preferred)
           Examples: "acetaminophen", "ibuprofen", "amoxicillin"
    
    Output: Summary of adverse events reported to FDA
    
    Note: This provides real FDA data. For information from our curated
    medical documents, use search_drug_database instead.
    
    Tip: Use generic drug names for better results.
         "acetaminophen" works better than "Tylenol"
         "ibuprofen" works better than "Advil"
    """
    print(f"\n   [Tool] Tool: get_fda_adverse_events('{drug_name}')")
    
    # Normalize drug name
    drug_name_lower = drug_name.lower().strip()
    
    # Map common names to FDA generic names
    drug_aliases = {
        "paracetamol": "acetaminophen",
        "tylenol": "acetaminophen",
        "advil": "ibuprofen",
        "motrin": "ibuprofen",
        "dolo": "acetaminophen",
        "crocin": "acetaminophen",
    }
    
    fda_drug_name = drug_aliases.get(drug_name_lower, drug_name_lower)
    print(f"   [FDA] Searching for: {fda_drug_name}")
    
    try:
        # Get reaction counts (aggregated data)
        counts_data = get_reaction_counts(fda_drug_name, top_n=10)
        
        # Get sample reports
        reports_data = query_fda_api(fda_drug_name, limit=3)
        
        # Build response
        parts = []
        parts.append(f"FDA Adverse Events for: {drug_name.upper()}")
        parts.append(f"(Searched as: {fda_drug_name})")
        parts.append("")
        
        # Section 1: Most reported reactions
        if counts_data and "results" in counts_data:
            parts.append("MOST REPORTED ADVERSE REACTIONS:")
            parts.append("(Based on FDA adverse event reports)")
            parts.append("")
            
            for item in counts_data["results"][:10]:
                reaction = item.get("term", "Unknown")
                count = item.get("count", 0)
                parts.append(f"  - {reaction}: {count:,} reports")
            
            parts.append("")
        else:
            parts.append("No aggregated reaction data available.")
            parts.append("")
        
        # Section 2: Sample report details
        if reports_data and "results" in reports_data:
            parts.append("SAMPLE ADVERSE EVENT REPORTS:")
            parts.append("")
            
            for i, report in enumerate(reports_data["results"][:2], 1):
                # Extract key information
                serious = report.get("serious", 0)
                serious_text = "SERIOUS" if serious == 1 else "Non-serious"
                
                # Get reactions
                reactions = []
                for reaction in report.get("patient", {}).get("reaction", [])[:5]:
                    reactions.append(reaction.get("reactionmeddrapt", "Unknown"))
                
                # Get outcome
                outcomes = report.get("patient", {}).get("patientdeath", None)
                outcome_text = "Death reported" if outcomes else "Non-fatal"
                
                parts.append(f"  Report {i}: {serious_text} | {outcome_text}")
                if reactions:
                    parts.append(f"    Reactions: {', '.join(reactions)}")
                parts.append("")
        
        # Handle no results
        if len(parts) <= 4:
            # Check if drug wasn't found
            if reports_data and reports_data.get("error") == "not_found":
                return (
                    f"No adverse event reports found for '{drug_name}' in FDA database.\n\n"
                    f"Tips:\n"
                    f"- Try the generic name instead of brand name\n"
                    f"- Check spelling\n"
                    f"- Common alternatives: 'acetaminophen' for Tylenol/Paracetamol, "
                    f"'ibuprofen' for Advil"
                )
            else:
                return f"Could not retrieve FDA data for '{drug_name}'. API may be temporarily unavailable."
        
        # Add disclaimer
        parts.append("-" * 40)
        parts.append("Data source: FDA Adverse Event Reporting System (FAERS)")
        parts.append("Note: Reports don't prove the drug caused the reaction.")
        parts.append("Always consult healthcare professionals for medical advice.")
        
        result = "\n".join(parts)
        print(f"   [Result] Retrieved FDA adverse event data for {fda_drug_name}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error querying FDA API: {str(e)}"
        print(f"   [Error] {error_msg}")
        return error_msg


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FDA ADVERSE EVENTS TOOL TEST")
    print("=" * 60)
    
    # Test with common drugs
    test_drugs = [
        "acetaminophen",
        "ibuprofen",
        "amoxicillin"
    ]
    
    for drug in test_drugs:
        print(f"\n{'='*60}")
        print(f"Drug: {drug}")
        print("=" * 60)
        
        result = get_fda_adverse_events.invoke(drug)
        print(f"\n{result}")
