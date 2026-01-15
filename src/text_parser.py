import re

# Conversion factors to standard units (grams for weight, ml for volume)
UNIT_CONVERSIONS = {
    # Weight -> grams
    'lb': 453.592, 'lbs': 453.592, 'pound': 453.592, 'pounds': 453.592,
    'oz': 28.3495, 'ounce': 28.3495, 'ounces': 28.3495,
    'kg': 1000.0, 'kilogram': 1000.0, 'kilograms': 1000.0,
    'g': 1.0, 'gram': 1.0, 'grams': 1.0,
    'mg': 0.001, 'milligram': 0.001,
    # Volume -> milliliters
    'l': 1000.0, 'liter': 1000.0, 'liters': 1000.0, 'litre': 1000.0,
    'ml': 1.0, 'milliliter': 1.0, 'milliliters': 1.0,
    'fl oz': 29.5735, 'fluid ounce': 29.5735, 'fl. oz': 29.5735
}

def extract_value_unit(text):
    # 1. Try explicit format "Value: ... Unit: ..."
    value_match = re.search(r"Value:\s*([\d\.]+)", text)
    unit_match = re.search(r"Unit:\s*([A-Za-z\s\.]+)", text)

    if value_match and unit_match:
        return float(value_match.group(1)), unit_match.group(1).strip().lower()

    # 2. Try natural language patterns (e.g., "12 oz", "5 lbs")
    # Pattern: Number + optional space + Unit
    pattern = r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)?)"
    matches = re.findall(pattern, text)
    
    # Iterate reversed to find the most likely relevant quantity (often at end of title/desc)
    for val_str, unit_str in reversed(matches):
        unit_clean = unit_str.strip().lower().rstrip('s') # simple singularization
        # Check against keys (handling plurals via map or stripping 's')
        if unit_clean != "mass" and unit_clean != "volume" and unit_clean != "size":
            if unit_clean in UNIT_CONVERSIONS or (unit_clean + 's') in UNIT_CONVERSIONS:
                return float(val_str), unit_clean
        else:
            return float(val_str), unit_clean
    return None, None

def normalize_to_standard_unit(value, unit):
    """Converts value to standard units (g or ml)."""
    if value is None or unit is None:
        return None
        
    unit = unit.lower().strip()
    # Handle variations like 'fl. oz' -> 'fl oz'
    unit = unit.replace('.', '')
    
    # Check direct match or singular form
    if unit in UNIT_CONVERSIONS:
        return value * UNIT_CONVERSIONS[unit]
    if unit.endswith('s') and unit[:-1] in UNIT_CONVERSIONS:
        return value * UNIT_CONVERSIONS[unit[:-1]]
        
    return None

def extract_pack_size(text):
    # Enhanced patterns for pack size
    patterns = [
        r"Pack of\s*(\d+)",
        r"(\d+)\s*pack",
        r"count\s*(\d+)",
        r"(\d+)\s*count",
        r"(\d+)\s*pcs"
    ]
    
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
            
    return 1

def compute_total_quantity(value, pack_size):
    if value is None:
        return None
    return value * pack_size
