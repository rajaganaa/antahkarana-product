"""
Tool 4: Medicine Expiry Checker
===============================

A Python computation tool that checks if medicine is expired.

This demonstrates a DATE COMPUTATION tool:
- Parses various date formats
- Calculates days remaining or past expiry
- Provides clear expired/valid status

Supports formats:
- "Dec 2025", "December 2025"
- "12/2025", "12-2025"
- "2025-12", "2025/12"
- "12/25" (assumes 20xx)
"""

from langchain.tools import tool
from datetime import datetime, date
import re
from typing import Optional, Tuple


# Month name mappings
MONTH_NAMES = {
    'jan': 1, 'january': 1,
    'feb': 2, 'february': 2,
    'mar': 3, 'march': 3,
    'apr': 4, 'april': 4,
    'may': 5,
    'jun': 6, 'june': 6,
    'jul': 7, 'july': 7,
    'aug': 8, 'august': 8,
    'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10,
    'nov': 11, 'november': 11,
    'dec': 12, 'december': 12,
}


def parse_expiry_date(date_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse expiry date string into (year, month) tuple.
    
    Supports various formats commonly found on medicine packaging.
    
    Returns:
        Tuple of (year, month) or None if parsing fails
    """
    date_str = date_str.strip().lower()
    
    # Pattern 1: "Dec 2025", "December 2025"
    match = re.match(r'([a-z]+)\s*(\d{4})', date_str)
    if match:
        month_name, year = match.groups()
        if month_name in MONTH_NAMES:
            return (int(year), MONTH_NAMES[month_name])
    
    # Pattern 2: "2025 Dec", "2025 December"
    match = re.match(r'(\d{4})\s*([a-z]+)', date_str)
    if match:
        year, month_name = match.groups()
        if month_name in MONTH_NAMES:
            return (int(year), MONTH_NAMES[month_name])
    
    # Pattern 3: "12/2025", "12-2025" (month/year)
    match = re.match(r'(\d{1,2})[/\-](\d{4})', date_str)
    if match:
        month, year = match.groups()
        month = int(month)
        if 1 <= month <= 12:
            return (int(year), month)
    
    # Pattern 4: "2025/12", "2025-12" (year/month)
    match = re.match(r'(\d{4})[/\-](\d{1,2})', date_str)
    if match:
        year, month = match.groups()
        month = int(month)
        if 1 <= month <= 12:
            return (int(year), month)
    
    # Pattern 5: "12/25" (month/year short) - assume 20xx
    match = re.match(r'(\d{1,2})[/\-](\d{2})$', date_str)
    if match:
        month, year = match.groups()
        month = int(month)
        year = int(year)
        if 1 <= month <= 12:
            # Assume 2000s
            year = 2000 + year if year < 100 else year
            return (year, month)
    
    # Pattern 6: Just year "2025" - assume December
    match = re.match(r'^(\d{4})$', date_str)
    if match:
        year = int(match.group(1))
        return (year, 12)
    
    return None


def get_expiry_date_end(year: int, month: int) -> date:
    """
    Get the last day of the expiry month.
    
    Medicine expiry typically means "good until end of this month".
    """
    # Move to next month, then subtract one day
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    
    # Last day of expiry month
    from datetime import timedelta
    return next_month - timedelta(days=1)


@tool
def check_medicine_expiry(expiry_date: str) -> str:
    """
    Check if a medicine is expired based on its expiry date.
    
    Use this tool when user asks about:
    - Whether medicine is expired
    - How long until medicine expires
    - If it's safe to use (regarding expiry)
    - Days remaining before expiry
    
    Input: Expiry date as shown on medicine package
           Supported formats:
           - "Dec 2025", "December 2025"
           - "12/2025", "12-2025"
           - "2025-12", "2025/12"
           - "12/25" (assumes 2025)
    
    Output: Expiry status with days remaining or days since expiry
    
    Note: This tool checks DATE only. For drug safety information,
    use search_drug_database or get_fda_adverse_events.
    """
    print(f"\n   [Tool] check_medicine_expiry('{expiry_date}')")
    
    # Parse the expiry date
    parsed = parse_expiry_date(expiry_date)
    
    if parsed is None:
        return (
            f"Could not parse expiry date: '{expiry_date}'\n\n"
            f"Please provide the date in one of these formats:\n"
            f"  - 'Dec 2025' or 'December 2025'\n"
            f"  - '12/2025' or '12-2025'\n"
            f"  - '2025-12' or '2025/12'\n"
        )
    
    year, month = parsed
    
    # Validate year is reasonable
    current_year = datetime.now().year
    if year < current_year - 10 or year > current_year + 20:
        return f"Year {year} seems unusual for an expiry date. Please verify."
    
    # Get the expiry date (end of month)
    expiry = get_expiry_date_end(year, month)
    today = date.today()
    
    # Calculate difference
    delta = expiry - today
    days_diff = delta.days
    
    # Format month name
    month_names = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_name = month_names[month]
    
    # Build response
    lines = []
    
    if days_diff < 0:
        # EXPIRED
        days_past = abs(days_diff)
        months_past = days_past // 30
        
        lines.append("[X] EXPIRED")
        lines.append("")
        lines.append(f"Expiry Date: {month_name} {year}")
        lines.append(f"Today: {today.strftime('%B %d, %Y')}")
        lines.append("")
        lines.append(f"Expired: {days_past} days ago ({months_past} months)")
        lines.append("")
        lines.append("[!] RECOMMENDATION:")
        lines.append("   Do NOT use this medicine. Expired medications may:")
        lines.append("   - Be less effective")
        lines.append("   - Have degraded into harmful compounds")
        lines.append("   - Cause unexpected side effects")
        lines.append("")
        lines.append("   Please dispose of properly and get a fresh supply.")
        
    elif days_diff <= 30:
        # EXPIRING SOON (within 30 days)
        lines.append("[!] EXPIRING SOON")
        lines.append("")
        lines.append(f"Expiry Date: {month_name} {year}")
        lines.append(f"Today: {today.strftime('%B %d, %Y')}")
        lines.append("")
        lines.append(f"Expires in: {days_diff} days")
        lines.append("")
        lines.append("RECOMMENDATION:")
        lines.append("   The medicine is still valid but will expire soon.")
        lines.append("   - Use before expiry date")
        lines.append("   - Consider getting a fresh supply for future use")
        lines.append("   - Do not start a new course if it may extend past expiry")
        
    elif days_diff <= 90:
        # VALID but expiring in 3 months
        months_remaining = days_diff // 30
        
        lines.append("[OK] VALID (Expiring in ~{} months)".format(months_remaining))
        lines.append("")
        lines.append(f"Expiry Date: {month_name} {year}")
        lines.append(f"Today: {today.strftime('%B %d, %Y')}")
        lines.append("")
        lines.append(f"Valid for: {days_diff} days ({months_remaining} months)")
        lines.append("")
        lines.append("The medicine is safe to use.")
        
    else:
        # VALID with plenty of time
        months_remaining = days_diff // 30
        years_remaining = days_diff // 365
        
        lines.append("[OK] VALID")
        lines.append("")
        lines.append(f"Expiry Date: {month_name} {year}")
        lines.append(f"Today: {today.strftime('%B %d, %Y')}")
        lines.append("")
        
        if years_remaining >= 1:
            lines.append(f"Valid for: {years_remaining} year(s) and {months_remaining % 12} month(s)")
        else:
            lines.append(f"Valid for: {months_remaining} months ({days_diff} days)")
        
        lines.append("")
        lines.append("The medicine is safe to use regarding expiry date.")
    
    # Add storage note
    lines.append("")
    lines.append("-" * 40)
    lines.append("Tip: Always store medicines as directed on the package")
    lines.append("   (usually in a cool, dry place away from sunlight).")
    
    result = "\n".join(lines)
    
    status = "EXPIRED" if days_diff < 0 else "VALID"
    print(f"   [Result] Status: {status} (diff: {days_diff} days)")
    
    return result


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXPIRY CHECKER TOOL TEST")
    print("=" * 60)
    
    # Test various date formats
    test_dates = [
        "Dec 2025",
        "December 2025",
        "12/2025",
        "2025-12",
        "06/24",        # Short format (June 2024 - likely expired)
        "March 2024",   # Past date
        "Jan 2030",     # Far future
        "Invalid Date", # Error case
    ]
    
    for date_str in test_dates:
        print(f"\n{'='*60}")
        print(f"Testing: '{date_str}'")
        print("=" * 60)
        
        result = check_medicine_expiry.invoke(date_str)
        print(f"\n{result}")
