SELECT
    child_key,
    DATE_TRUNC('month', event_datetime) AS index_month,
    MIN(event_datetime) AS first_event_datetime,
    MAX(event_datetime) AS last_event_datetime,
    MAX(patient_age_in_days) AS age_days,
    MAX(patient_sex) AS sex
FROM slv_iz_events
GROUP BY 1, 2;
