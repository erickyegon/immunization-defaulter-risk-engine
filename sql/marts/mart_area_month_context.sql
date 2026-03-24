SELECT
    county_clean,
    sub_county_clean,
    registry_snapshot_month AS index_month,
    COUNT(DISTINCT chw_key) AS active_chws
FROM slv_chw_registry
GROUP BY 1, 2, 3;
