from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VaccineScheduleRule:
    vaccine_name: str
    due_age_days: int


KENYA_CHILD_SCHEDULE = [
    VaccineScheduleRule("has_bcg", 0),
    VaccineScheduleRule("has_opv_0", 0),
    VaccineScheduleRule("has_opv_1", 42),
    VaccineScheduleRule("has_penta_1", 42),
    VaccineScheduleRule("has_pcv_1", 42),
    VaccineScheduleRule("has_rota_1", 42),
    VaccineScheduleRule("has_opv_2", 70),
    VaccineScheduleRule("has_penta_2", 70),
    VaccineScheduleRule("has_pcv_2", 70),
    VaccineScheduleRule("has_rota_2", 70),
    VaccineScheduleRule("has_opv_3", 98),
    VaccineScheduleRule("has_penta_3", 98),
    VaccineScheduleRule("has_pcv_3", 98),
    VaccineScheduleRule("has_ipv", 98),
    VaccineScheduleRule("has_measles_9_months", 270),
]


def expected_vaccines_by_age(age_days: float | int | None) -> list[str]:
    if age_days is None:
        return []
    return [rule.vaccine_name for rule in KENYA_CHILD_SCHEDULE if age_days >= rule.due_age_days]
