SELECT
    family_id AS family_key,
    chw_uuid AS chw_key,
    chw_area,
    reported AS homevisit_datetime,
    record_hash
FROM brz_homevisit;
