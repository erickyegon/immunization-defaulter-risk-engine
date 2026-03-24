SELECT
    chw_uuid AS chw_key,
    chw_name,
    chw_area_uuid AS chw_area_key,
    chw_area_name,
    community_unit,
    county_name,
    sub_county_name,
    reported AS registry_snapshot_datetime,
    record_hash
FROM brz_active_chps;
