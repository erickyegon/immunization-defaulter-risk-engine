SELECT
    uuid AS supervision_event_key,
    chw_uuid AS chw_key,
    reported AS supervision_datetime,
    last_visit_date,
    calc_assessment_score,
    calc_immunization_score,
    has_all_tools,
    has_proper_protective_equipment,
    has_essential_medicines,
    record_hash
FROM brz_supervision;
