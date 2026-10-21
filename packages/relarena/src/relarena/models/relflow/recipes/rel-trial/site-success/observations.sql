-- Feature-only query: labels remain in the runner-owned anchor pipeline.
WITH
anchors AS (
    SELECT "__row_position", "__entity_key", "__as_of"
    FROM read_parquet($anchors)
    WHERE "__row_position" >= $start AND "__row_position" < $stop
),
root_features AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "as_of",
        root."facility_id" AS "entity_id",
        root."name" AS "facility_name",
        root."city" AS "city",
        root."state" AS "state",
        root."zip" AS "postal_code",
        root."country" AS "country"
    FROM anchors AS anchor
    LEFT JOIN "facilities" AS root
        ON anchor."__entity_key" = root."facility_id"
    QUALIFY row_number() OVER (
        PARTITION BY anchor."__row_position"
        ORDER BY root."__source_row_id" DESC NULLS LAST
    ) = 1
),
"studies_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        event."nct_id" AS "study_id",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "facilities_studies" AS event
        ON anchor."__entity_key" = event."facility_id"
        AND event."date" <= anchor."__as_of"
),
"studies_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "study_id" := "study_id"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "studies"
    FROM "studies_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("studies_grouped"."studies", []) AS "studies"
FROM root_features
LEFT JOIN "studies_grouped" USING ("__row_position")
ORDER BY "__row_position";
