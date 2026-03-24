SELECT
    uuid AS iz_event_key,
    patient_id AS child_key,
    reported AS event_datetime,
    patient_sex,
    patient_age_in_days,
    patient_age_in_months,
    patient_age_in_years,
    county,
    month,
    record_hash,
    source_id
FROM brz_iz;
