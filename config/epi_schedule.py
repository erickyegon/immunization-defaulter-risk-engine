"""
config/epi_schedule.py
─────────────────────────────────────────────────────────────────────────────
Kenya Expanded Programme on Immunization schedule as Python constants.
Used by the feature engineering pipeline for age-gated vaccine completeness.
"""

# Core vaccines (non-endemic — always expected): (column_name, min_age_months)
CORE_VACCINE_SCHEDULE = [
    ("has_bcg",               0),
    ("has_opv_0",             0),
    ("has_opv_1",             2),   # 6 weeks ≈ 1.5 months → use 2
    ("has_opv_2",             3),   # 10 weeks ≈ 2.5 months → use 3
    ("has_opv_3",             4),   # 14 weeks ≈ 3.5 months → use 4
    ("has_pcv_1",             2),
    ("has_pcv_2",             3),
    ("has_pcv_3",             4),
    ("has_penta_1",           2),
    ("has_penta_2",           3),
    ("has_penta_3",           4),
    ("has_ipv",               4),
    ("has_rota_1",            2),
    ("has_rota_2",            3),
    ("has_rota_3",            4),
    ("has_measles_9_months",  9),
    ("has_measles_18_months",18),
]

# Malaria vaccine (R21/Matrix-M) — endemic regions only
MALARIA_VACCINE_SCHEDULE = [
    ("has_malaria_6_vaccine",  5),
    ("has_malaria_7_vaccine",  6),
    ("has_malaria_8_vaccine",  7),
    ("has_malaria_24_vaccine",24),
]

CORE_VACCINES   = [v[0] for v in CORE_VACCINE_SCHEDULE]
MALARIA_VACCINES = [v[0] for v in MALARIA_VACCINE_SCHEDULE]
ALL_VACCINES    = CORE_VACCINES + MALARIA_VACCINES

# Vitamin A schedule: (column_name, min_age_months)
VITAMIN_A_SCHEDULE = [
    ("has_vitamin_a_6_months",  6),
    ("has_vitamin_a_12_months",12),
    ("has_vitamin_a_18_months",18),
    ("has_vitamin_a_24_months",24),
    ("has_vitamin_a_30_months",30),
    ("has_vitamin_a_36_months",36),
    ("has_vitamin_a_42_months",42),
    ("has_vitamin_a_48_months",48),
    ("has_vitamin_a_54_months",54),
    ("has_vitamin_a_60_months",60),
]
VITAMIN_A_COLS = [v[0] for v in VITAMIN_A_SCHEDULE]


def get_expected_vaccines(age_months: float, malaria_endemic: bool = False) -> int:
    """
    Return the count of vaccines a child is expected to have received
    by a given age (in months) according to the Kenya EPI schedule.

    Parameters
    ----------
    age_months : float
        Child's age in months
    malaria_endemic : bool
        Whether the child lives in a malaria endemic region

    Returns
    -------
    int
        Number of expected vaccine doses
    """
    if age_months is None or age_months != age_months:  # NaN check
        return 0

    count = sum(1 for _, min_age in CORE_VACCINE_SCHEDULE if age_months >= min_age)

    if malaria_endemic:
        count += sum(1 for _, min_age in MALARIA_VACCINE_SCHEDULE if age_months >= min_age)

    return count


def get_expected_vitamin_a(age_months: float) -> int:
    """Count of Vitamin A doses expected by a given age."""
    if age_months is None or age_months != age_months:
        return 0
    return sum(1 for _, min_age in VITAMIN_A_SCHEDULE if age_months >= min_age)


# Human-readable labels for SHAP explanations
VACCINE_LABELS = {
    "has_bcg":               "BCG",
    "has_opv_0":             "OPV-0 (Birth Polio)",
    "has_opv_1":             "OPV-1",
    "has_opv_2":             "OPV-2",
    "has_opv_3":             "OPV-3",
    "has_pcv_1":             "PCV-1",
    "has_pcv_2":             "PCV-2",
    "has_pcv_3":             "PCV-3",
    "has_penta_1":           "Penta-1 (DPT-HepB-Hib)",
    "has_penta_2":           "Penta-2",
    "has_penta_3":           "Penta-3",
    "has_ipv":               "IPV",
    "has_rota_1":            "Rota-1",
    "has_rota_2":            "Rota-2",
    "has_rota_3":            "Rota-3",
    "has_measles_9_months":  "Measles-Rubella MR1 (9m)",
    "has_measles_18_months": "Measles-Rubella MR2 (18m)",
    "has_malaria_6_vaccine": "Malaria Dose 1 (R21)",
    "has_malaria_7_vaccine": "Malaria Dose 2 (R21)",
    "has_malaria_8_vaccine": "Malaria Dose 3 (R21)",
    "has_malaria_24_vaccine":"Malaria Booster (R21)",
}
