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
        root."driverId" AS "entity_id",
        root."driverRef" AS "driver_ref",
        root."code" AS "code",
        root."forename" AS "forename",
        root."surname" AS "surname",
        root."dob" AS "date_of_birth",
        root."nationality" AS "nationality"
    FROM anchors AS anchor
    LEFT JOIN "drivers" AS root
        ON anchor."__entity_key" = root."driverId"
    QUALIFY row_number() OVER (
        PARTITION BY anchor."__row_position"
        ORDER BY root."__source_row_id" DESC NULLS LAST
    ) = 1
),
"results_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        event."resultId" AS "result_id",
        event."date" AS "race_date",
        event."raceId" AS "race_id",
        event."constructorId" AS "constructor_id",
        event."number" AS "car_number",
        event."grid" AS "grid",
        event."position" AS "finish_position",
        event."positionOrder" AS "position_order",
        event."points" AS "points",
        event."laps" AS "laps",
        event."milliseconds" AS "milliseconds",
        event."fastestLap" AS "fastest_lap",
        event."rank" AS "fastest_lap_rank",
        event."statusId" AS "status_id",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "results" AS event
        ON anchor."__entity_key" = event."driverId"
        AND event."date" <= anchor."__as_of"
),
"results_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "result_id" := "result_id",
                "race_date" := "race_date",
                "race_id" := "race_id",
                "constructor_id" := "constructor_id",
                "car_number" := "car_number",
                "grid" := "grid",
                "finish_position" := "finish_position",
                "position_order" := "position_order",
                "points" := "points",
                "laps" := "laps",
                "milliseconds" := "milliseconds",
                "fastest_lap" := "fastest_lap",
                "fastest_lap_rank" := "fastest_lap_rank",
                "status_id" := "status_id"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "results"
    FROM "results_ranked"
    WHERE "__recency" <= 128
    GROUP BY "__row_position"
),
"qualifying_results_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        event."qualifyId" AS "qualifying_id",
        event."date" AS "qualifying_date",
        event."raceId" AS "race_id",
        event."constructorId" AS "constructor_id",
        event."number" AS "car_number",
        event."position" AS "position",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "qualifying" AS event
        ON anchor."__entity_key" = event."driverId"
        AND event."date" <= anchor."__as_of"
),
"qualifying_results_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "qualifying_id" := "qualifying_id",
                "qualifying_date" := "qualifying_date",
                "race_id" := "race_id",
                "constructor_id" := "constructor_id",
                "car_number" := "car_number",
                "position" := "position"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "qualifying_results"
    FROM "qualifying_results_ranked"
    WHERE "__recency" <= 128
    GROUP BY "__row_position"
),
"standings_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        event."driverStandingsId" AS "standing_id",
        event."date" AS "standing_date",
        event."raceId" AS "race_id",
        event."points" AS "points",
        event."position" AS "position",
        event."wins" AS "wins",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "standings" AS event
        ON anchor."__entity_key" = event."driverId"
        AND event."date" <= anchor."__as_of"
),
"standings_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "standing_id" := "standing_id",
                "standing_date" := "standing_date",
                "race_id" := "race_id",
                "points" := "points",
                "position" := "position",
                "wins" := "wins"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "standings"
    FROM "standings_ranked"
    WHERE "__recency" <= 128
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("results_grouped"."results", []) AS "results",
    coalesce("qualifying_results_grouped"."qualifying_results", []) AS "qualifying_results",
    coalesce("standings_grouped"."standings", []) AS "standings"
FROM root_features
LEFT JOIN "results_grouped" USING ("__row_position")
LEFT JOIN "qualifying_results_grouped" USING ("__row_position")
LEFT JOIN "standings_grouped" USING ("__row_position")
ORDER BY "__row_position";
