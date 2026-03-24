SELECT
    chw_key,
    DATE_TRUNC('month', supervision_datetime) AS index_month,
    COUNT(*) AS supervision_events,
    AVG(calc_immunization_score) AS mean_immunization_score
FROM slv_supervision_events
GROUP BY 1, 2;
