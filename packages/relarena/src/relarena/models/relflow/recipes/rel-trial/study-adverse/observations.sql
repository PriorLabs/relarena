-- Labels remain in runner-owned anchors. Timestamped rows obey the scoring cutoff.
-- Static attributes follow the benchmark snapshot; historical attribute versions are not supplied.
-- Raw history caps do not restrict the full-history summaries.
WITH
anchors AS (
    SELECT "__row_position", "__entity_key", "__as_of"
    FROM read_parquet($anchors)
    WHERE "__row_position" >= $start AND "__row_position" < $stop
),
root_features AS (
    SELECT anchor."__row_position",
        anchor."__as_of" AS "as_of",
        extract(year FROM anchor."__as_of") AS "as_of_year",
        root."nct_id" IS NOT NULL AS "entity_available",
        CASE WHEN root."nct_id" IS NOT NULL THEN 'rel-trial:study:' || CAST(root."nct_id" AS VARCHAR) END AS "entity_id",
        root."start_date" AS "start_date",
        root."target_duration" AS "target_duration",
        root."study_type" AS "study_type",
        CASE WHEN root."acronym" IS NOT NULL THEN 'rel-trial:acronym:' || CAST(root."acronym" AS VARCHAR) END AS "acronym",
        root."baseline_population" AS "baseline_population",
        root."brief_title" AS "brief_title",
        root."official_title" AS "official_title",
        root."phase" AS "phase",
        root."enrollment" AS "enrollment",
        root."enrollment_type" AS "enrollment_type",
        root."source" AS "source",
        CAST(root."limitations_and_caveats" AS VARCHAR) AS "limitations_and_caveats",
        root."source_class" AS "source_class",
        root."number_of_arms" AS "number_of_arms",
        root."number_of_groups" AS "number_of_groups",
        root."has_dmc" AS "has_dmc",
        root."is_fda_regulated_drug" AS "regulated_drug",
        root."is_fda_regulated_device" AS "regulated_device",
        root."is_unapproved_device" AS "unapproved_device",
        root."is_ppsd" AS "ppsd",
        root."is_us_export" AS "us_export",
        root."biospec_retention" AS "biospec_retention",
        root."biospec_description" AS "biospec_description",
        root."baseline_type_units_analyzed" AS "baseline_type_units_analyzed",
        root."fdaaa801_violation" AS "fdaaa801_violation",
        root."plan_to_share_ipd" AS "plan_to_share_ipd",
        root."detailed_descriptions" AS "detailed_description",
        root."brief_summaries" AS "brief_summary",
        date_diff('microseconds', root."start_date", anchor."__as_of") / 86400000000.0 AS "study_age_days",
        CASE WHEN regexp_full_match(lower(trim(CAST(root."target_duration" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)') THEN TRY_CAST(regexp_extract(lower(trim(CAST(root."target_duration" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 1) AS DOUBLE) * CASE WHEN regexp_extract(lower(trim(CAST(root."target_duration" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'day%' THEN 1.0 WHEN regexp_extract(lower(trim(CAST(root."target_duration" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'week%' THEN 7.0 WHEN regexp_extract(lower(trim(CAST(root."target_duration" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'month%' THEN 30.4375 WHEN regexp_extract(lower(trim(CAST(root."target_duration" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'year%' THEN 365.25 WHEN regexp_extract(lower(trim(CAST(root."target_duration" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'hour%' THEN 1.0 / 24 END END AS "duration_days"
    FROM anchors AS anchor
    LEFT JOIN "studies" AS root ON anchor."__entity_key" = root."nct_id" AND root."start_date" <= anchor."__as_of"
    QUALIFY row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY root."start_date" DESC NULLS LAST, root."__source_row_id" DESC NULLS LAST) = 1
),
adverse_events_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        event."event_type" AS "event_type",
        event."classification" AS "classification",
        TRY_CAST(event."subjects_affected" AS DOUBLE) AS "subjects_affected",
        TRY_CAST(event."subjects_at_risk" AS DOUBLE) AS "subjects_at_risk",
        TRY_CAST(event."subjects_affected" AS DOUBLE) / nullif(TRY_CAST(event."subjects_at_risk" AS DOUBLE), 0) AS "affected_rate",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "reported_event_totals" AS event ON anchor."__entity_key" = event."nct_id" AND event."date" <= anchor."__as_of"
),
adverse_events_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "event_type" := "event_type",
                "classification" := "classification",
                "subjects_affected" := "subjects_affected",
                "subjects_at_risk" := "subjects_at_risk",
                "affected_rate" := "affected_rate"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "adverse_events"
    FROM adverse_events_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
adverse_events_summary AS (
    SELECT "__row_position",
        count(*) AS "adverse_events_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "adverse_events_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "adverse_events_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "adverse_events_count_365d",
        min("age_days") AS "adverse_events_last_age_days",
        count(DISTINCT "event_type") AS "adverse_events_distinct_event_type",
        count(DISTINCT "classification") AS "adverse_events_distinct_classification",
        count("subjects_affected") AS "adverse_events_subjects_affected_known_count",
        CASE WHEN count("subjects_affected") = count(*) THEN sum("subjects_affected") END AS "adverse_events_subjects_affected_sum",
        count("subjects_at_risk") AS "adverse_events_subjects_at_risk_known_count",
        CASE WHEN count("subjects_at_risk") = count(*) THEN sum("subjects_at_risk") END AS "adverse_events_subjects_at_risk_sum"
    FROM adverse_events_ranked
    GROUP BY "__row_position"
),
outcomes_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."id" IS NOT NULL THEN 'rel-trial:outcome:' || CAST(event."id" AS VARCHAR) END AS "outcome_id",
        event."outcome_type" AS "outcome_type",
        event."units" AS "units",
        event."units_analyzed" AS "units_analyzed",
        event."dispersion_type" AS "dispersion_type",
        event."param_type" AS "parameter_type",
        event."title" AS "title",
        event."description" AS "outcome_description",
        event."time_frame" AS "time_frame",
        event."population" AS "population",
        TRY_CAST(analysis."p_value" AS DOUBLE) AS "analysis_p_value",
        TRY_CAST(analysis."param_value" AS DOUBLE) AS "analysis_param_value",
        TRY_CAST(analysis."dispersion_value" AS DOUBLE) AS "analysis_dispersion_value",
        TRY_CAST(analysis."ci_percent" AS DOUBLE) AS "analysis_ci_percent",
        TRY_CAST(analysis."ci_lower_limit" AS DOUBLE) AS "analysis_ci_lower_limit",
        TRY_CAST(analysis."ci_upper_limit" AS DOUBLE) AS "analysis_ci_upper_limit",
        analysis."method" AS "analysis_method",
        date_diff('microseconds', analysis."date", anchor."__as_of") / 86400000000.0 AS "analysis_age_days",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "outcomes" AS event ON anchor."__entity_key" = event."nct_id" AND event."date" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "outcome_analyses" AS linked WHERE linked."outcome_id" = event."id" AND linked."date" <= anchor."__as_of" AND linked."nct_id" = event."nct_id" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS analysis ON TRUE
),
outcomes_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "outcome_id" := "outcome_id",
                "outcome_type" := "outcome_type",
                "units" := "units",
                "units_analyzed" := "units_analyzed",
                "dispersion_type" := "dispersion_type",
                "parameter_type" := "parameter_type",
                "title" := "title",
                "outcome_description" := "outcome_description",
                "time_frame" := "time_frame",
                "population" := "population",
                "analysis_p_value" := "analysis_p_value",
                "analysis_param_value" := "analysis_param_value",
                "analysis_dispersion_value" := "analysis_dispersion_value",
                "analysis_ci_percent" := "analysis_ci_percent",
                "analysis_ci_lower_limit" := "analysis_ci_lower_limit",
                "analysis_ci_upper_limit" := "analysis_ci_upper_limit",
                "analysis_method" := "analysis_method",
                "analysis_age_days" := "analysis_age_days"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "outcomes"
    FROM outcomes_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
outcomes_summary AS (
    SELECT "__row_position",
        count(*) AS "outcomes_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "outcomes_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "outcomes_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "outcomes_count_365d",
        min("age_days") AS "outcomes_last_age_days",
        count(DISTINCT "outcome_id") AS "outcomes_distinct_outcome_id",
        count(DISTINCT "outcome_type") AS "outcomes_distinct_outcome_type"
    FROM outcomes_ranked
    GROUP BY "__row_position"
),
designs_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        event."allocation" AS "allocation",
        event."intervention_model" AS "intervention_model",
        event."observational_model" AS "observational_model",
        event."primary_purpose" AS "primary_purpose",
        event."time_perspective" AS "time_perspective",
        event."masking" AS "masking",
        event."subject_masked" AS "subject_masked",
        event."caregiver_masked" AS "caregiver_masked",
        event."investigator_masked" AS "investigator_masked",
        event."outcomes_assessor_masked" AS "outcomes_assessor_masked",
        event."masking_description" AS "masking_description",
        event."intervention_model_description" AS "intervention_model_description",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "designs" AS event ON anchor."__entity_key" = event."nct_id" AND event."date" <= anchor."__as_of"
),
designs_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "masking_description" := "masking_description",
                "intervention_model_description" := "intervention_model_description",
                "age_days" := "age_days",
                "allocation" := "allocation",
                "intervention_model" := "intervention_model",
                "observational_model" := "observational_model",
                "primary_purpose" := "primary_purpose",
                "time_perspective" := "time_perspective",
                "masking" := "masking",
                "subject_masked" := "subject_masked",
                "caregiver_masked" := "caregiver_masked",
                "investigator_masked" := "investigator_masked",
                "outcomes_assessor_masked" := "outcomes_assessor_masked"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "designs"
    FROM designs_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
designs_summary AS (
    SELECT "__row_position",
        count(*) AS "designs_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "designs_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "designs_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "designs_count_365d",
        min("age_days") AS "designs_last_age_days"
    FROM designs_ranked
    GROUP BY "__row_position"
),
eligibilities_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        event."sampling_method" AS "sampling_method",
        event."gender" AS "gender",
        event."minimum_age" AS "minimum_age",
        event."maximum_age" AS "maximum_age",
        event."healthy_volunteers" AS "healthy_volunteers",
        event."adult" AS "adult",
        event."child" AS "child",
        event."older_adult" AS "older_adult",
        CASE WHEN regexp_full_match(lower(trim(CAST(event."minimum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)') THEN TRY_CAST(regexp_extract(lower(trim(CAST(event."minimum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 1) AS DOUBLE) * CASE WHEN regexp_extract(lower(trim(CAST(event."minimum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'day%' THEN 1.0 WHEN regexp_extract(lower(trim(CAST(event."minimum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'week%' THEN 7.0 WHEN regexp_extract(lower(trim(CAST(event."minimum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'month%' THEN 30.4375 WHEN regexp_extract(lower(trim(CAST(event."minimum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'year%' THEN 365.25 WHEN regexp_extract(lower(trim(CAST(event."minimum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'hour%' THEN 1.0 / 24 END END AS "minimum_age_days",
        CASE WHEN regexp_full_match(lower(trim(CAST(event."maximum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)') THEN TRY_CAST(regexp_extract(lower(trim(CAST(event."maximum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 1) AS DOUBLE) * CASE WHEN regexp_extract(lower(trim(CAST(event."maximum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'day%' THEN 1.0 WHEN regexp_extract(lower(trim(CAST(event."maximum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'week%' THEN 7.0 WHEN regexp_extract(lower(trim(CAST(event."maximum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'month%' THEN 30.4375 WHEN regexp_extract(lower(trim(CAST(event."maximum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'year%' THEN 365.25 WHEN regexp_extract(lower(trim(CAST(event."maximum_age" AS VARCHAR))), '([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?|hours?)', 2) LIKE 'hour%' THEN 1.0 / 24 END END AS "maximum_age_days",
        event."population" AS "population",
        event."criteria" AS "criteria",
        event."gender_description" AS "gender_description",
        event."gender_based" AS "gender_based",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "eligibilities" AS event ON anchor."__entity_key" = event."nct_id" AND event."date" <= anchor."__as_of"
),
eligibilities_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "gender_description" := "gender_description",
                "gender_based" := "gender_based",
                "age_days" := "age_days",
                "sampling_method" := "sampling_method",
                "gender" := "gender",
                "minimum_age" := "minimum_age",
                "maximum_age" := "maximum_age",
                "healthy_volunteers" := "healthy_volunteers",
                "adult" := "adult",
                "child" := "child",
                "older_adult" := "older_adult",
                "minimum_age_days" := "minimum_age_days",
                "maximum_age_days" := "maximum_age_days",
                "population" := "population",
                "criteria" := "criteria"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "eligibilities"
    FROM eligibilities_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
eligibilities_summary AS (
    SELECT "__row_position",
        count(*) AS "eligibilities_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "eligibilities_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "eligibilities_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "eligibilities_count_365d",
        min("age_days") AS "eligibilities_last_age_days"
    FROM eligibilities_ranked
    GROUP BY "__row_position"
),
conditions_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."condition_id" IS NOT NULL THEN 'rel-trial:condition:' || CAST(event."condition_id" AS VARCHAR) END AS "condition_id",
        details."mesh_term" AS "term",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "conditions_studies" AS event ON anchor."__entity_key" = event."nct_id" AND event."date" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "conditions" AS linked WHERE linked."condition_id" = event."condition_id" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS details ON TRUE
),
conditions_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "condition_id" := "condition_id",
                "term" := "term"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "conditions"
    FROM conditions_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
conditions_summary AS (
    SELECT "__row_position",
        count(*) AS "conditions_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "conditions_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "conditions_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "conditions_count_365d",
        min("age_days") AS "conditions_last_age_days",
        count(DISTINCT "condition_id") AS "conditions_distinct_condition_id"
    FROM conditions_ranked
    GROUP BY "__row_position"
),
interventions_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."intervention_id" IS NOT NULL THEN 'rel-trial:intervention:' || CAST(event."intervention_id" AS VARCHAR) END AS "intervention_id",
        details."mesh_term" AS "term",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "interventions_studies" AS event ON anchor."__entity_key" = event."nct_id" AND event."date" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "interventions" AS linked WHERE linked."intervention_id" = event."intervention_id" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS details ON TRUE
),
interventions_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "intervention_id" := "intervention_id",
                "term" := "term"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "interventions"
    FROM interventions_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
interventions_summary AS (
    SELECT "__row_position",
        count(*) AS "interventions_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "interventions_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "interventions_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "interventions_count_365d",
        min("age_days") AS "interventions_last_age_days",
        count(DISTINCT "intervention_id") AS "interventions_distinct_intervention_id"
    FROM interventions_ranked
    GROUP BY "__row_position"
),
sponsors_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."sponsor_id" IS NOT NULL THEN 'rel-trial:sponsor:' || CAST(event."sponsor_id" AS VARCHAR) END AS "sponsor_id",
        event."lead_or_collaborator" AS "role",
        details."name" AS "name",
        details."agency_class" AS "agency_class",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "sponsors_studies" AS event ON anchor."__entity_key" = event."nct_id" AND event."date" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "sponsors" AS linked WHERE linked."sponsor_id" = event."sponsor_id" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS details ON TRUE
),
sponsors_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "sponsor_id" := "sponsor_id",
                "role" := "role",
                "name" := "name",
                "agency_class" := "agency_class"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "sponsors"
    FROM sponsors_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
sponsors_summary AS (
    SELECT "__row_position",
        count(*) AS "sponsors_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "sponsors_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "sponsors_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "sponsors_count_365d",
        min("age_days") AS "sponsors_last_age_days",
        count(DISTINCT "sponsor_id") AS "sponsors_distinct_sponsor_id"
    FROM sponsors_ranked
    GROUP BY "__row_position"
),
facilities_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."facility_id" IS NOT NULL THEN 'rel-trial:facility:' || CAST(event."facility_id" AS VARCHAR) END AS "facility_id",
        details."name" AS "name",
        details."city" AS "city",
        details."state" AS "state",
        CASE WHEN details."zip" IS NOT NULL THEN 'rel-trial:postal:' || CAST(details."zip" AS VARCHAR) END AS "postal_code",
        details."country" AS "country",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "facilities_studies" AS event ON anchor."__entity_key" = event."nct_id" AND event."date" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "facilities" AS linked WHERE linked."facility_id" = event."facility_id" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS details ON TRUE
),
facilities_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "facility_id" := "facility_id",
                "name" := "name",
                "city" := "city",
                "state" := "state",
                "postal_code" := "postal_code",
                "country" := "country"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "facilities"
    FROM facilities_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
facilities_summary AS (
    SELECT "__row_position",
        count(*) AS "facilities_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "facilities_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "facilities_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "facilities_count_365d",
        min("age_days") AS "facilities_last_age_days",
        count(DISTINCT "facility_id") AS "facilities_distinct_facility_id",
        count(DISTINCT "country") AS "facilities_distinct_country"
    FROM facilities_ranked
    GROUP BY "__row_position"
),
-- Peer trials are deduplicated per relationship kind before aggregating past reports.
-- The 256 most recently completed eligible peers per kind bound preparation work;
-- candidate and retained counts expose this information limit explicitly.
root_studies AS (
    SELECT DISTINCT anchor."__row_position", anchor."__as_of", study."nct_id"
    FROM anchors AS anchor
    INNER JOIN "studies" AS study ON study."nct_id" = anchor."__entity_key" AND study."start_date" <= anchor."__as_of"
),
trial_memberships AS (
    SELECT "nct_id", "date", 'condition' AS "relation", CAST("condition_id" AS VARCHAR) AS "related_id" FROM "conditions_studies" WHERE "condition_id" IS NOT NULL UNION ALL SELECT "nct_id", "date", 'intervention' AS "relation", CAST("intervention_id" AS VARCHAR) AS "related_id" FROM "interventions_studies" WHERE "intervention_id" IS NOT NULL UNION ALL SELECT "nct_id", "date", 'sponsor' AS "relation", CAST("sponsor_id" AS VARCHAR) AS "related_id" FROM "sponsors_studies" WHERE "sponsor_id" IS NOT NULL UNION ALL SELECT "nct_id", "date", 'facility' AS "relation", CAST("facility_id" AS VARCHAR) AS "related_id" FROM "facilities_studies" WHERE "facility_id" IS NOT NULL
),
peer_candidates AS (
    SELECT DISTINCT root."__row_position", root."__as_of", peer."relation", peer."nct_id"
    FROM root_studies AS root
    INNER JOIN trial_memberships AS own ON own."nct_id" = root."nct_id" AND own."date" <= root."__as_of"
    INNER JOIN trial_memberships AS peer ON peer."relation" = own."relation" AND peer."related_id" = own."related_id"
        AND peer."date" <= root."__as_of" AND peer."nct_id" != root."nct_id"
),
completion_reports AS (
    SELECT "nct_id", "date" FROM "reported_event_totals"
    UNION ALL SELECT "nct_id", "date" FROM "outcomes"
    UNION ALL SELECT "nct_id", "date" FROM "outcome_analyses"
    UNION ALL SELECT "nct_id", "date" FROM "drop_withdrawals"
),
peer_completed AS (
    SELECT peer."__row_position", peer."__as_of", peer."relation", peer."nct_id", max(report."date") AS "completion_date"
    FROM peer_candidates AS peer
    INNER JOIN completion_reports AS report ON report."nct_id" = peer."nct_id" AND report."date" <= peer."__as_of"
    GROUP BY peer."__row_position", peer."__as_of", peer."relation", peer."nct_id"
),
peer_ranked AS (
    SELECT peer.*, TRY_CAST(study."enrollment" AS DOUBLE) AS "enrollment",
        count(*) OVER (PARTITION BY peer."__row_position", peer."relation") AS "candidate_studies",
        row_number() OVER (PARTITION BY peer."__row_position", peer."relation" ORDER BY peer."completion_date" DESC, peer."nct_id") AS "__peer_rank"
    FROM peer_completed AS peer
    INNER JOIN LATERAL (SELECT linked.* FROM "studies" AS linked WHERE linked."nct_id" = peer."nct_id" AND linked."start_date" <= peer."__as_of" ORDER BY linked."start_date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS study ON TRUE
),
selected_peers AS (
    SELECT * FROM peer_ranked WHERE "__peer_rank" <= 256
),
peer_reports AS (
    SELECT peer.*, adverse."report_count" AS "adverse_report_count", adverse."subjects_affected", adverse."subjects_at_risk",
        primary_reports."report_count" AS "primary_report_count", primary_reports."any_success", primary_reports."all_success",
        withdrawals."report_count" AS "withdrawal_report_count", withdrawals."withdrawal_count",
        peer."completion_date" AS "last_report_date"
    FROM selected_peers AS peer
    LEFT JOIN LATERAL (
        SELECT count(*) AS "report_count", max(event."date") AS "last_date",
            CASE WHEN count(TRY_CAST(event."subjects_affected" AS DOUBLE)) = count(*) THEN sum(TRY_CAST(event."subjects_affected" AS DOUBLE)) END AS "subjects_affected",
            CASE WHEN count(TRY_CAST(event."subjects_at_risk" AS DOUBLE)) = count(*) THEN sum(TRY_CAST(event."subjects_at_risk" AS DOUBLE)) END AS "subjects_at_risk"
        FROM "reported_event_totals" AS event
        WHERE event."nct_id" = peer."nct_id" AND event."date" <= peer."__as_of" AND CAST(event."event_type" AS VARCHAR) IN ('serious', 'deaths')
    ) AS adverse ON TRUE
    LEFT JOIN LATERAL (
        SELECT count(*) AS "report_count", max(analysis."date") AS "last_date",
            max(CASE WHEN TRY_CAST(analysis."p_value" AS DOUBLE) <= 0.05 THEN 1.0 ELSE 0.0 END) AS "any_success",
            min(CASE WHEN TRY_CAST(analysis."p_value" AS DOUBLE) < 0.05 THEN 1.0 ELSE 0.0 END) AS "all_success"
        FROM "outcome_analyses" AS analysis
        WHERE analysis."nct_id" = peer."nct_id" AND analysis."date" <= peer."__as_of"
            AND (analysis."p_value_modifier" IS NULL OR CAST(analysis."p_value_modifier" AS VARCHAR) != '>')
            AND TRY_CAST(analysis."p_value" AS DOUBLE) BETWEEN 0 AND 1
            AND EXISTS (SELECT 1 FROM "outcomes" AS outcome WHERE outcome."id" = analysis."outcome_id" AND outcome."nct_id" = analysis."nct_id" AND outcome."date" <= peer."__as_of" AND CAST(outcome."outcome_type" AS VARCHAR) = 'Primary')
    ) AS primary_reports ON TRUE
    LEFT JOIN LATERAL (
        SELECT count(*) AS "report_count", max(event."date") AS "last_date",
            CASE WHEN count(TRY_CAST(event."count" AS DOUBLE)) = count(*) THEN sum(TRY_CAST(event."count" AS DOUBLE)) END AS "withdrawal_count"
        FROM "drop_withdrawals" AS event WHERE event."nct_id" = peer."nct_id" AND event."date" <= peer."__as_of"
    ) AS withdrawals ON TRUE
),
peer_summary AS (
    SELECT "__row_position", "relation", max("candidate_studies") AS "candidate_studies", count(*) AS "retained_studies",
        count(*) FILTER (WHERE "last_report_date" IS NOT NULL) AS "completed_studies",
        count(*) FILTER (WHERE "last_report_date" >= "__as_of" - INTERVAL '365 days') AS "completed_studies_365d",
        count(*) FILTER (WHERE "adverse_report_count" > 0) AS "adverse_report_studies",
        count("subjects_affected") AS "adverse_affected_known_studies", avg("subjects_affected") AS "adverse_affected_mean", median("subjects_affected") AS "adverse_affected_median",
        avg("subjects_at_risk") AS "subjects_at_risk_mean", count(*) FILTER (WHERE "primary_report_count" > 0) AS "primary_report_studies",
        avg("any_success") AS "primary_any_success_rate", avg("all_success") AS "primary_all_success_rate",
        count(*) FILTER (WHERE "withdrawal_report_count" > 0) AS "withdrawal_report_studies", count("withdrawal_count") AS "withdrawal_count_known_studies",
        avg("withdrawal_count") AS "withdrawal_count_mean", min(date_diff('microseconds', "last_report_date", "__as_of") / 86400000000.0) AS "last_report_age_days",
        avg("enrollment") AS "enrollment_mean"
    FROM peer_reports GROUP BY "__row_position", "relation"
),
peer_context AS (
    SELECT "__row_position", list(struct_pack("relation" := "relation", "candidate_studies" := "candidate_studies", "retained_studies" := "retained_studies", "completed_studies" := "completed_studies", "completed_studies_365d" := "completed_studies_365d", "adverse_report_studies" := "adverse_report_studies", "adverse_affected_known_studies" := "adverse_affected_known_studies", "adverse_affected_mean" := "adverse_affected_mean", "adverse_affected_median" := "adverse_affected_median", "subjects_at_risk_mean" := "subjects_at_risk_mean", "primary_report_studies" := "primary_report_studies", "primary_any_success_rate" := "primary_any_success_rate", "primary_all_success_rate" := "primary_all_success_rate", "withdrawal_report_studies" := "withdrawal_report_studies", "withdrawal_count_known_studies" := "withdrawal_count_known_studies", "withdrawal_count_mean" := "withdrawal_count_mean", "last_report_age_days" := "last_report_age_days", "enrollment_mean" := "enrollment_mean") ORDER BY "relation") AS "peer_context"
    FROM peer_summary GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce(peer_context."peer_context", []) AS "peer_context",
    coalesce(adverse_events_grouped."adverse_events", []) AS "adverse_events",
    coalesce(adverse_events_summary."adverse_events_count", 0) AS "adverse_events_count",
    coalesce(adverse_events_summary."adverse_events_count_30d", 0) AS "adverse_events_count_30d",
    coalesce(adverse_events_summary."adverse_events_count_90d", 0) AS "adverse_events_count_90d",
    coalesce(adverse_events_summary."adverse_events_count_365d", 0) AS "adverse_events_count_365d",
    adverse_events_summary."adverse_events_last_age_days" AS "adverse_events_last_age_days",
    coalesce(adverse_events_summary."adverse_events_distinct_event_type", 0) AS "adverse_events_distinct_event_type",
    coalesce(adverse_events_summary."adverse_events_distinct_classification", 0) AS "adverse_events_distinct_classification",
    coalesce(adverse_events_summary."adverse_events_subjects_affected_known_count", 0) AS "adverse_events_subjects_affected_known_count",
    CASE WHEN adverse_events_summary."__row_position" IS NULL THEN 0 ELSE adverse_events_summary."adverse_events_subjects_affected_sum" END AS "adverse_events_subjects_affected_sum",
    coalesce(adverse_events_summary."adverse_events_subjects_at_risk_known_count", 0) AS "adverse_events_subjects_at_risk_known_count",
    CASE WHEN adverse_events_summary."__row_position" IS NULL THEN 0 ELSE adverse_events_summary."adverse_events_subjects_at_risk_sum" END AS "adverse_events_subjects_at_risk_sum",
    coalesce(outcomes_grouped."outcomes", []) AS "outcomes",
    coalesce(outcomes_summary."outcomes_count", 0) AS "outcomes_count",
    coalesce(outcomes_summary."outcomes_count_30d", 0) AS "outcomes_count_30d",
    coalesce(outcomes_summary."outcomes_count_90d", 0) AS "outcomes_count_90d",
    coalesce(outcomes_summary."outcomes_count_365d", 0) AS "outcomes_count_365d",
    outcomes_summary."outcomes_last_age_days" AS "outcomes_last_age_days",
    coalesce(outcomes_summary."outcomes_distinct_outcome_id", 0) AS "outcomes_distinct_outcome_id",
    coalesce(outcomes_summary."outcomes_distinct_outcome_type", 0) AS "outcomes_distinct_outcome_type",
    coalesce(designs_grouped."designs", []) AS "designs",
    coalesce(designs_summary."designs_count", 0) AS "designs_count",
    coalesce(designs_summary."designs_count_30d", 0) AS "designs_count_30d",
    coalesce(designs_summary."designs_count_90d", 0) AS "designs_count_90d",
    coalesce(designs_summary."designs_count_365d", 0) AS "designs_count_365d",
    designs_summary."designs_last_age_days" AS "designs_last_age_days",
    coalesce(eligibilities_grouped."eligibilities", []) AS "eligibilities",
    coalesce(eligibilities_summary."eligibilities_count", 0) AS "eligibilities_count",
    coalesce(eligibilities_summary."eligibilities_count_30d", 0) AS "eligibilities_count_30d",
    coalesce(eligibilities_summary."eligibilities_count_90d", 0) AS "eligibilities_count_90d",
    coalesce(eligibilities_summary."eligibilities_count_365d", 0) AS "eligibilities_count_365d",
    eligibilities_summary."eligibilities_last_age_days" AS "eligibilities_last_age_days",
    coalesce(conditions_grouped."conditions", []) AS "conditions",
    coalesce(conditions_summary."conditions_count", 0) AS "conditions_count",
    coalesce(conditions_summary."conditions_count_30d", 0) AS "conditions_count_30d",
    coalesce(conditions_summary."conditions_count_90d", 0) AS "conditions_count_90d",
    coalesce(conditions_summary."conditions_count_365d", 0) AS "conditions_count_365d",
    conditions_summary."conditions_last_age_days" AS "conditions_last_age_days",
    coalesce(conditions_summary."conditions_distinct_condition_id", 0) AS "conditions_distinct_condition_id",
    coalesce(interventions_grouped."interventions", []) AS "interventions",
    coalesce(interventions_summary."interventions_count", 0) AS "interventions_count",
    coalesce(interventions_summary."interventions_count_30d", 0) AS "interventions_count_30d",
    coalesce(interventions_summary."interventions_count_90d", 0) AS "interventions_count_90d",
    coalesce(interventions_summary."interventions_count_365d", 0) AS "interventions_count_365d",
    interventions_summary."interventions_last_age_days" AS "interventions_last_age_days",
    coalesce(interventions_summary."interventions_distinct_intervention_id", 0) AS "interventions_distinct_intervention_id",
    coalesce(sponsors_grouped."sponsors", []) AS "sponsors",
    coalesce(sponsors_summary."sponsors_count", 0) AS "sponsors_count",
    coalesce(sponsors_summary."sponsors_count_30d", 0) AS "sponsors_count_30d",
    coalesce(sponsors_summary."sponsors_count_90d", 0) AS "sponsors_count_90d",
    coalesce(sponsors_summary."sponsors_count_365d", 0) AS "sponsors_count_365d",
    sponsors_summary."sponsors_last_age_days" AS "sponsors_last_age_days",
    coalesce(sponsors_summary."sponsors_distinct_sponsor_id", 0) AS "sponsors_distinct_sponsor_id",
    coalesce(facilities_grouped."facilities", []) AS "facilities",
    coalesce(facilities_summary."facilities_count", 0) AS "facilities_count",
    coalesce(facilities_summary."facilities_count_30d", 0) AS "facilities_count_30d",
    coalesce(facilities_summary."facilities_count_90d", 0) AS "facilities_count_90d",
    coalesce(facilities_summary."facilities_count_365d", 0) AS "facilities_count_365d",
    facilities_summary."facilities_last_age_days" AS "facilities_last_age_days",
    coalesce(facilities_summary."facilities_distinct_facility_id", 0) AS "facilities_distinct_facility_id",
    coalesce(facilities_summary."facilities_distinct_country", 0) AS "facilities_distinct_country"
FROM root_features
LEFT JOIN adverse_events_grouped USING ("__row_position")
LEFT JOIN adverse_events_summary USING ("__row_position")
LEFT JOIN outcomes_grouped USING ("__row_position")
LEFT JOIN outcomes_summary USING ("__row_position")
LEFT JOIN designs_grouped USING ("__row_position")
LEFT JOIN designs_summary USING ("__row_position")
LEFT JOIN eligibilities_grouped USING ("__row_position")
LEFT JOIN eligibilities_summary USING ("__row_position")
LEFT JOIN conditions_grouped USING ("__row_position")
LEFT JOIN conditions_summary USING ("__row_position")
LEFT JOIN interventions_grouped USING ("__row_position")
LEFT JOIN interventions_summary USING ("__row_position")
LEFT JOIN sponsors_grouped USING ("__row_position")
LEFT JOIN sponsors_summary USING ("__row_position")
LEFT JOIN facilities_grouped USING ("__row_position")
LEFT JOIN facilities_summary USING ("__row_position")
LEFT JOIN peer_context USING ("__row_position")
ORDER BY "__row_position";
