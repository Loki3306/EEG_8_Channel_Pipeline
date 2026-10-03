# Authoritative 64-Channel DTU Mapping
DTU_CHANNELS = {
    0: "FP1", 1: "AF7", 2: "AF3", 3: "F1", 4: "F3", 5: "F5", 6: "F7", 7: "FT7", 8: "FC5",
    9: "FC3", 10: "FC1", 11: "C1", 12: "C3", 13: "C5", 14: "T7", 15: "TP7", 16: "CP5",
    17: "CP3", 18: "CP1", 19: "P1", 20: "P3", 21: "P5", 22: "P7", 23: "P9", 24: "PO7",
    25: "PO3", 26: "O1", 27: "IZ", 28: "OZ", 29: "POZ", 30: "PZ", 31: "CPZ", 32: "FPZ",
    33: "FP2", 34: "AF8", 35: "AF4", 36: "AFZ", 37: "FZ", 38: "F2", 39: "F4", 40: "F6",
    41: "F8", 42: "FT8", 43: "FC6", 44: "FC4", 45: "FC2", 46: "FCZ", 47: "CZ", 48: "C2",
    49: "C4", 50: "C6", 51: "T8", 52: "TP8", 53: "CP6", 54: "CP4", 55: "CP2", 56: "P2",
    57: "P4", 58: "P6", 59: "P8", 60: "P10", 61: "PO8", 62: "PO4", 63: "O2"
}

# Reverse mapping for easy lookup
NAME_TO_IDX = {name: idx for idx, name in DTU_CHANNELS.items()}

def get_indices(*names):
    return [NAME_TO_IDX[n.upper()] for n in names]

# ==========================================
# TIER 2: ANATOMICAL MONTAGES
# ==========================================
TIER2_FRONTAL = get_indices("FP1", "FP2", "F7", "F8", "F3", "F4", "F1", "F2")
TIER2_FRONTO_TEMPORAL = get_indices("F7", "F8", "FT7", "FT8", "T7", "T8", "FC5", "FC6")
TIER2_TEMPORAL = get_indices("T7", "T8", "TP7", "TP8", "FT7", "FT8", "P7", "P8")
TIER2_CENTRAL = get_indices("C1", "C2", "C3", "C4", "C5", "C6", "CZ", "FCZ")
TIER2_PARIETAL = get_indices("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")
TIER2_POSTERIOR = get_indices("PO7", "PO8", "PO3", "PO4", "O1", "O2", "POZ", "OZ")

# Bilateral contrasts
TIER2_BILATERAL_TEMPORAL = get_indices("T7", "T8", "TP7", "TP8", "P7", "P8", "PO7", "PO8")
TIER2_BILATERAL_CENTRAL = get_indices("F3", "F4", "FC3", "FC4", "C3", "C4", "CP3", "CP4")

# ==========================================
# TIER 3: NEAR-EAR MONTAGES
# ==========================================
# Strict near-ear (immediately around the pinna)
TIER3_NEAR_EAR_STRICT = get_indices("T7", "T8", "TP7", "TP8", "FT7", "FT8", "P7", "P8")

# Expanded near-ear (next ring out)
TIER3_NEAR_EAR_EXPANDED = get_indices("T7", "T8", "TP7", "TP8", "CP5", "CP6", "FC5", "FC6")

# Ear + Temporal bias
TIER3_NEAR_EAR_TEMPORAL = get_indices("FT7", "FT8", "T7", "T8", "TP7", "TP8", "CP5", "CP6")

# ==========================================
# ALL DEFINED MONTAGES DICT
# ==========================================
MONTAGES = {
    "frontal": TIER2_FRONTAL,
    "fronto_temporal": TIER2_FRONTO_TEMPORAL,
    "temporal": TIER2_TEMPORAL,
    "central": TIER2_CENTRAL,
    "parietal": TIER2_PARIETAL,
    "posterior": TIER2_POSTERIOR,
    "bilateral_temporal": TIER2_BILATERAL_TEMPORAL,
    "bilateral_central": TIER2_BILATERAL_CENTRAL,
    "near_ear_strict": TIER3_NEAR_EAR_STRICT,
    "near_ear_expanded": TIER3_NEAR_EAR_EXPANDED,
    "near_ear_temporal": TIER3_NEAR_EAR_TEMPORAL
}
