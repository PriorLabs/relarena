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
        root."driverId" IS NOT NULL AS "entity_available",
        CASE WHEN root."driverId" IS NOT NULL THEN 'rel-f1:driver:' || CAST(root."driverId" AS VARCHAR) END AS "entity_id",
        CASE WHEN root."driverRef" IS NOT NULL THEN 'rel-f1:driver_ref:' || CAST(root."driverRef" AS VARCHAR) END AS "driver_ref",
        CASE WHEN root."code" IS NOT NULL THEN 'rel-f1:driver_code:' || CAST(root."code" AS VARCHAR) END AS "code",
        root."forename" AS "forename",
        root."surname" AS "surname",
        root."dob" AS "date_of_birth",
        date_diff('microseconds', root."dob", anchor."__as_of") / 86400000000.0 AS "driver_age_days",
        root."nationality" AS "nationality"
    FROM anchors AS anchor
    LEFT JOIN "drivers" AS root ON anchor."__entity_key" = root."driverId"
    QUALIFY row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY root."__source_row_id" DESC NULLS LAST) = 1
),
results_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."resultId" IS NOT NULL THEN 'rel-f1:result:' || CAST(event."resultId" AS VARCHAR) END AS "result_id",
        event."date" AS "race_date",
        CASE WHEN event."raceId" IS NOT NULL THEN 'rel-f1:race:' || CAST(event."raceId" AS VARCHAR) END AS "race_id",
        event."constructorId" AS "constructor_id",
        race."circuitId" AS "circuit_id",
        event."number" AS "car_number",
        TRY_CAST(event."grid" AS DOUBLE) AS "grid",
        TRY_CAST(event."position" AS DOUBLE) AS "finish_position",
        TRY_CAST(event."positionOrder" AS DOUBLE) AS "position_order",
        TRY_CAST(event."points" AS DOUBLE) AS "points",
        TRY_CAST(event."laps" AS DOUBLE) AS "laps",
        TRY_CAST(event."milliseconds" AS DOUBLE) AS "milliseconds",
        TRY_CAST(event."fastestLap" AS DOUBLE) AS "fastest_lap",
        TRY_CAST(event."rank" AS DOUBLE) AS "fastest_lap_rank",
        event."statusId" AS "status_id",
        TRY_CAST(qualifying."position" AS DOUBLE) AS "qualifying_position",
        TRY_CAST(standing."position" AS DOUBLE) AS "standing_position",
        TRY_CAST(standing."points" AS DOUBLE) AS "standing_points",
        TRY_CAST(standing."wins" AS DOUBLE) AS "standing_wins",
        qualifying."__source_row_id" IS NOT NULL AS "qualifying_available",
        standing."__source_row_id" IS NOT NULL AS "standing_available",
        TRY_CAST(race."year" AS DOUBLE) AS "race_year",
        TRY_CAST(race."round" AS DOUBLE) AS "race_round",
        race."name" AS "race_name",
        CAST(race."time" AS VARCHAR) AS "race_time",
        circuit."circuitRef" AS "circuit_ref",
        circuit."name" AS "circuit_name",
        circuit."location" AS "circuit_location",
        circuit."country" AS "circuit_country",
        TRY_CAST(circuit."lat" AS DOUBLE) AS "circuit_lat",
        TRY_CAST(circuit."lng" AS DOUBLE) AS "circuit_lng",
        TRY_CAST(circuit."alt" AS DOUBLE) AS "circuit_alt",
        constructor."constructorRef" AS "constructor_ref",
        constructor."name" AS "constructor_name",
        constructor."nationality" AS "constructor_nationality",
        TRY_CAST(team_result."points" AS DOUBLE) AS "constructor_race_points",
        TRY_CAST(team_standing."points" AS DOUBLE) AS "constructor_standing_points",
        TRY_CAST(team_standing."position" AS DOUBLE) AS "constructor_standing_position",
        TRY_CAST(team_standing."wins" AS DOUBLE) AS "constructor_standing_wins",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "results" AS event ON anchor."__entity_key" = event."driverId" AND event."date" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "races" AS linked WHERE linked."raceId" = event."raceId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS race ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "qualifying" AS linked WHERE linked."raceId" = event."raceId" AND linked."date" <= anchor."__as_of" AND linked."driverId" = event."driverId" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS qualifying ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "standings" AS linked WHERE linked."raceId" = event."raceId" AND linked."date" <= anchor."__as_of" AND linked."driverId" = event."driverId" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS standing ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "circuits" AS linked WHERE linked."circuitId" = race."circuitId" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS circuit ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "constructors" AS linked WHERE linked."constructorId" = event."constructorId" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS constructor ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "constructor_results" AS linked WHERE linked."raceId" = event."raceId" AND linked."constructorId" = event."constructorId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS team_result ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "constructor_standings" AS linked WHERE linked."raceId" = event."raceId" AND linked."constructorId" = event."constructorId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS team_standing ON TRUE
),
results_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "race_year" := "race_year",
                "race_round" := "race_round",
                "race_name" := "race_name",
                "race_time" := "race_time",
                "circuit_ref" := "circuit_ref",
                "circuit_name" := "circuit_name",
                "circuit_location" := "circuit_location",
                "circuit_country" := "circuit_country",
                "circuit_lat" := "circuit_lat",
                "circuit_lng" := "circuit_lng",
                "circuit_alt" := "circuit_alt",
                "constructor_ref" := "constructor_ref",
                "constructor_name" := "constructor_name",
                "constructor_nationality" := "constructor_nationality",
                "constructor_race_points" := "constructor_race_points",
                "constructor_standing_points" := "constructor_standing_points",
                "constructor_standing_position" := "constructor_standing_position",
                "constructor_standing_wins" := "constructor_standing_wins",
                "age_days" := "age_days",
                "result_id" := "result_id",
                "race_date" := "race_date",
                "race_id" := "race_id",
                "constructor_id" := "constructor_id",
                "circuit_id" := "circuit_id",
                "car_number" := "car_number",
                "grid" := "grid",
                "finish_position" := "finish_position",
                "position_order" := "position_order",
                "points" := "points",
                "laps" := "laps",
                "milliseconds" := "milliseconds",
                "fastest_lap" := "fastest_lap",
                "fastest_lap_rank" := "fastest_lap_rank",
                "status_id" := "status_id",
                "qualifying_position" := "qualifying_position",
                "standing_position" := "standing_position",
                "standing_points" := "standing_points",
                "standing_wins" := "standing_wins",
                "qualifying_available" := "qualifying_available",
                "standing_available" := "standing_available"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "results"
    FROM results_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
results_summary AS (
    SELECT "__row_position",
        count(*) AS "results_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "results_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "results_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "results_count_365d",
        min("age_days") AS "results_last_age_days",
        count(DISTINCT "race_id") AS "results_distinct_race_id",
        count(DISTINCT "constructor_id") AS "results_distinct_constructor_id",
        count(DISTINCT "circuit_id") AS "results_distinct_circuit_id",
        count("points") AS "results_points_known_count",
        CASE WHEN count("points") = count(*) THEN sum("points") END AS "results_points_sum",
        count("laps") AS "results_laps_known_count",
        CASE WHEN count("laps") = count(*) THEN sum("laps") END AS "results_laps_sum",
        count("position_order") AS "results_position_order_known_count",
        avg("position_order") AS "results_position_order_mean",
        count("grid") AS "results_grid_known_count",
        avg("grid") AS "results_grid_mean",
        count("fastest_lap_rank") AS "results_fastest_lap_rank_known_count",
        avg("fastest_lap_rank") AS "results_fastest_lap_rank_mean"
    FROM results_ranked
    GROUP BY "__row_position"
),
qualifying_results_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."qualifyId" IS NOT NULL THEN 'rel-f1:qualifying:' || CAST(event."qualifyId" AS VARCHAR) END AS "qualifying_id",
        event."date" AS "qualifying_date",
        CASE WHEN event."raceId" IS NOT NULL THEN 'rel-f1:race:' || CAST(event."raceId" AS VARCHAR) END AS "race_id",
        event."constructorId" AS "constructor_id",
        race."circuitId" AS "circuit_id",
        event."number" AS "car_number",
        TRY_CAST(event."position" AS DOUBLE) AS "position",
        TRY_CAST(race."year" AS DOUBLE) AS "race_year",
        TRY_CAST(race."round" AS DOUBLE) AS "race_round",
        race."name" AS "race_name",
        CAST(race."time" AS VARCHAR) AS "race_time",
        circuit."circuitRef" AS "circuit_ref",
        circuit."name" AS "circuit_name",
        circuit."location" AS "circuit_location",
        circuit."country" AS "circuit_country",
        TRY_CAST(circuit."lat" AS DOUBLE) AS "circuit_lat",
        TRY_CAST(circuit."lng" AS DOUBLE) AS "circuit_lng",
        TRY_CAST(circuit."alt" AS DOUBLE) AS "circuit_alt",
        constructor."constructorRef" AS "constructor_ref",
        constructor."name" AS "constructor_name",
        constructor."nationality" AS "constructor_nationality",
        TRY_CAST(team_result."points" AS DOUBLE) AS "constructor_race_points",
        TRY_CAST(team_standing."points" AS DOUBLE) AS "constructor_standing_points",
        TRY_CAST(team_standing."position" AS DOUBLE) AS "constructor_standing_position",
        TRY_CAST(team_standing."wins" AS DOUBLE) AS "constructor_standing_wins",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "qualifying" AS event ON anchor."__entity_key" = event."driverId" AND event."date" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "races" AS linked WHERE linked."raceId" = event."raceId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS race ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "circuits" AS linked WHERE linked."circuitId" = race."circuitId" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS circuit ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "constructors" AS linked WHERE linked."constructorId" = event."constructorId" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS constructor ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "constructor_results" AS linked WHERE linked."raceId" = event."raceId" AND linked."constructorId" = event."constructorId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS team_result ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "constructor_standings" AS linked WHERE linked."raceId" = event."raceId" AND linked."constructorId" = event."constructorId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS team_standing ON TRUE
),
qualifying_results_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "race_year" := "race_year",
                "race_round" := "race_round",
                "race_name" := "race_name",
                "race_time" := "race_time",
                "circuit_ref" := "circuit_ref",
                "circuit_name" := "circuit_name",
                "circuit_location" := "circuit_location",
                "circuit_country" := "circuit_country",
                "circuit_lat" := "circuit_lat",
                "circuit_lng" := "circuit_lng",
                "circuit_alt" := "circuit_alt",
                "constructor_ref" := "constructor_ref",
                "constructor_name" := "constructor_name",
                "constructor_nationality" := "constructor_nationality",
                "constructor_race_points" := "constructor_race_points",
                "constructor_standing_points" := "constructor_standing_points",
                "constructor_standing_position" := "constructor_standing_position",
                "constructor_standing_wins" := "constructor_standing_wins",
                "age_days" := "age_days",
                "qualifying_id" := "qualifying_id",
                "qualifying_date" := "qualifying_date",
                "race_id" := "race_id",
                "constructor_id" := "constructor_id",
                "circuit_id" := "circuit_id",
                "car_number" := "car_number",
                "position" := "position"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "qualifying_results"
    FROM qualifying_results_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
qualifying_results_summary AS (
    SELECT "__row_position",
        count(*) AS "qualifying_results_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "qualifying_results_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "qualifying_results_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "qualifying_results_count_365d",
        min("age_days") AS "qualifying_results_last_age_days",
        count(DISTINCT "race_id") AS "qualifying_results_distinct_race_id",
        count(DISTINCT "constructor_id") AS "qualifying_results_distinct_constructor_id",
        count("position") AS "qualifying_results_position_known_count",
        avg("position") AS "qualifying_results_position_mean"
    FROM qualifying_results_ranked
    GROUP BY "__row_position"
),
standings_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."date" AS "__event_time",
        date_diff('microseconds', event."date", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."driverStandingsId" IS NOT NULL THEN 'rel-f1:standing:' || CAST(event."driverStandingsId" AS VARCHAR) END AS "standing_id",
        event."date" AS "standing_date",
        CASE WHEN event."raceId" IS NOT NULL THEN 'rel-f1:race:' || CAST(event."raceId" AS VARCHAR) END AS "race_id",
        race."circuitId" AS "circuit_id",
        TRY_CAST(event."points" AS DOUBLE) AS "points",
        TRY_CAST(event."position" AS DOUBLE) AS "position",
        TRY_CAST(event."wins" AS DOUBLE) AS "wins",
        TRY_CAST(race."year" AS DOUBLE) AS "race_year",
        TRY_CAST(race."round" AS DOUBLE) AS "race_round",
        race."name" AS "race_name",
        CAST(race."time" AS VARCHAR) AS "race_time",
        circuit."circuitRef" AS "circuit_ref",
        circuit."name" AS "circuit_name",
        circuit."location" AS "circuit_location",
        circuit."country" AS "circuit_country",
        TRY_CAST(circuit."lat" AS DOUBLE) AS "circuit_lat",
        TRY_CAST(circuit."lng" AS DOUBLE) AS "circuit_lng",
        TRY_CAST(circuit."alt" AS DOUBLE) AS "circuit_alt",
        constructor."constructorRef" AS "constructor_ref",
        constructor."name" AS "constructor_name",
        constructor."nationality" AS "constructor_nationality",
        TRY_CAST(team_result."points" AS DOUBLE) AS "constructor_race_points",
        TRY_CAST(team_standing."points" AS DOUBLE) AS "constructor_standing_points",
        TRY_CAST(team_standing."position" AS DOUBLE) AS "constructor_standing_position",
        TRY_CAST(team_standing."wins" AS DOUBLE) AS "constructor_standing_wins",
        driver_result."constructorId" AS "constructor_id",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "standings" AS event ON anchor."__entity_key" = event."driverId" AND event."date" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM "races" AS linked WHERE linked."raceId" = event."raceId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS race ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "results" AS linked WHERE linked."raceId" = event."raceId" AND linked."driverId" = event."driverId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS driver_result ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "circuits" AS linked WHERE linked."circuitId" = race."circuitId" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS circuit ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "constructors" AS linked WHERE linked."constructorId" = driver_result."constructorId" ORDER BY linked."__source_row_id" DESC LIMIT 1) AS constructor ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "constructor_results" AS linked WHERE linked."raceId" = event."raceId" AND linked."constructorId" = driver_result."constructorId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS team_result ON TRUE
    LEFT JOIN LATERAL (SELECT linked.* FROM "constructor_standings" AS linked WHERE linked."raceId" = event."raceId" AND linked."constructorId" = driver_result."constructorId" AND linked."date" <= anchor."__as_of" ORDER BY linked."date" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS team_standing ON TRUE
),
standings_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "race_year" := "race_year",
                "race_round" := "race_round",
                "race_name" := "race_name",
                "race_time" := "race_time",
                "circuit_ref" := "circuit_ref",
                "circuit_name" := "circuit_name",
                "circuit_location" := "circuit_location",
                "circuit_country" := "circuit_country",
                "circuit_lat" := "circuit_lat",
                "circuit_lng" := "circuit_lng",
                "circuit_alt" := "circuit_alt",
                "constructor_ref" := "constructor_ref",
                "constructor_name" := "constructor_name",
                "constructor_nationality" := "constructor_nationality",
                "constructor_race_points" := "constructor_race_points",
                "constructor_standing_points" := "constructor_standing_points",
                "constructor_standing_position" := "constructor_standing_position",
                "constructor_standing_wins" := "constructor_standing_wins",
                "constructor_id" := "constructor_id",
                "age_days" := "age_days",
                "standing_id" := "standing_id",
                "standing_date" := "standing_date",
                "race_id" := "race_id",
                "circuit_id" := "circuit_id",
                "points" := "points",
                "position" := "position",
                "wins" := "wins"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "standings"
    FROM standings_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
standings_summary AS (
    SELECT "__row_position",
        count(*) AS "standings_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "standings_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "standings_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "standings_count_365d",
        min("age_days") AS "standings_last_age_days",
        count(DISTINCT "race_id") AS "standings_distinct_race_id",
        count("position") AS "standings_position_known_count",
        avg("position") AS "standings_position_mean"
    FROM standings_ranked
    GROUP BY "__row_position"
),
constructor_summary AS (
    SELECT root."__row_position", latest."constructor_id" AS "latest_constructor_id",
        count(team."__source_row_id") AS "constructor_results_count",
        count(TRY_CAST(team."positionOrder" AS DOUBLE)) AS "constructor_position_known_count",
        avg(TRY_CAST(team."positionOrder" AS DOUBLE)) AS "constructor_position_mean",
        count(TRY_CAST(team."points" AS DOUBLE)) AS "constructor_points_known_count",
        CASE WHEN count(team."__source_row_id") = 0 THEN 0
             WHEN count(TRY_CAST(team."points" AS DOUBLE)) = count(team."__source_row_id")
             THEN sum(TRY_CAST(team."points" AS DOUBLE)) END AS "constructor_points_sum"
    FROM root_features AS root
    LEFT JOIN results_ranked AS latest ON latest."__row_position" = root."__row_position" AND latest."__recency" = 1
    LEFT JOIN "results" AS team ON team."constructorId" = latest."constructor_id" AND team."date" <= root."as_of"
    GROUP BY root."__row_position", latest."constructor_id"
),
constructor_race_summary AS (
    SELECT root."__row_position", races."race_count" AS "constructor_race_results_count",
        races."points_known_count" AS "constructor_race_points_known_count",
        races."points_sum" AS "constructor_race_points_sum",
        races."points_mean" AS "constructor_race_points_mean",
        TRY_CAST(standing."points" AS DOUBLE) AS "latest_constructor_standing_points",
        TRY_CAST(standing."position" AS DOUBLE) AS "latest_constructor_standing_position",
        TRY_CAST(standing."wins" AS DOUBLE) AS "latest_constructor_standing_wins"
    FROM root_features AS root
    LEFT JOIN results_ranked AS latest ON latest."__row_position" = root."__row_position" AND latest."__recency" = 1
    LEFT JOIN LATERAL (
        SELECT count(*) AS "race_count", count(TRY_CAST(event."points" AS DOUBLE)) AS "points_known_count",
            CASE WHEN count(*) = 0 THEN 0 WHEN count(TRY_CAST(event."points" AS DOUBLE)) = count(*) THEN sum(TRY_CAST(event."points" AS DOUBLE)) END AS "points_sum",
            avg(TRY_CAST(event."points" AS DOUBLE)) AS "points_mean"
        FROM "constructor_results" AS event
        WHERE event."constructorId" = latest."constructor_id" AND event."date" <= root."as_of"
    ) AS races ON TRUE
    LEFT JOIN LATERAL (SELECT event.* FROM "constructor_standings" AS event WHERE event."constructorId" = latest."constructor_id" AND event."date" <= root."as_of" ORDER BY event."date" DESC NULLS LAST, event."__source_row_id" DESC LIMIT 1) AS standing ON TRUE
)
SELECT
    root_features.*,
    constructor_race_summary."constructor_race_results_count" AS "constructor_race_results_count",
    constructor_race_summary."constructor_race_points_known_count" AS "constructor_race_points_known_count",
    constructor_race_summary."constructor_race_points_sum" AS "constructor_race_points_sum",
    constructor_race_summary."constructor_race_points_mean" AS "constructor_race_points_mean",
    constructor_race_summary."latest_constructor_standing_points" AS "latest_constructor_standing_points",
    constructor_race_summary."latest_constructor_standing_position" AS "latest_constructor_standing_position",
    constructor_race_summary."latest_constructor_standing_wins" AS "latest_constructor_standing_wins",

    coalesce(results_grouped."results", []) AS "results",
    coalesce(results_summary."results_count", 0) AS "results_count",
    coalesce(results_summary."results_count_30d", 0) AS "results_count_30d",
    coalesce(results_summary."results_count_90d", 0) AS "results_count_90d",
    coalesce(results_summary."results_count_365d", 0) AS "results_count_365d",
    results_summary."results_last_age_days" AS "results_last_age_days",
    coalesce(results_summary."results_distinct_race_id", 0) AS "results_distinct_race_id",
    coalesce(results_summary."results_distinct_constructor_id", 0) AS "results_distinct_constructor_id",
    coalesce(results_summary."results_distinct_circuit_id", 0) AS "results_distinct_circuit_id",
    coalesce(results_summary."results_points_known_count", 0) AS "results_points_known_count",
    CASE WHEN results_summary."__row_position" IS NULL THEN 0 ELSE results_summary."results_points_sum" END AS "results_points_sum",
    coalesce(results_summary."results_laps_known_count", 0) AS "results_laps_known_count",
    CASE WHEN results_summary."__row_position" IS NULL THEN 0 ELSE results_summary."results_laps_sum" END AS "results_laps_sum",
    coalesce(results_summary."results_position_order_known_count", 0) AS "results_position_order_known_count",
    results_summary."results_position_order_mean" AS "results_position_order_mean",
    coalesce(results_summary."results_grid_known_count", 0) AS "results_grid_known_count",
    results_summary."results_grid_mean" AS "results_grid_mean",
    coalesce(results_summary."results_fastest_lap_rank_known_count", 0) AS "results_fastest_lap_rank_known_count",
    results_summary."results_fastest_lap_rank_mean" AS "results_fastest_lap_rank_mean",
    coalesce(qualifying_results_grouped."qualifying_results", []) AS "qualifying_results",
    coalesce(qualifying_results_summary."qualifying_results_count", 0) AS "qualifying_results_count",
    coalesce(qualifying_results_summary."qualifying_results_count_30d", 0) AS "qualifying_results_count_30d",
    coalesce(qualifying_results_summary."qualifying_results_count_90d", 0) AS "qualifying_results_count_90d",
    coalesce(qualifying_results_summary."qualifying_results_count_365d", 0) AS "qualifying_results_count_365d",
    qualifying_results_summary."qualifying_results_last_age_days" AS "qualifying_results_last_age_days",
    coalesce(qualifying_results_summary."qualifying_results_distinct_race_id", 0) AS "qualifying_results_distinct_race_id",
    coalesce(qualifying_results_summary."qualifying_results_distinct_constructor_id", 0) AS "qualifying_results_distinct_constructor_id",
    coalesce(qualifying_results_summary."qualifying_results_position_known_count", 0) AS "qualifying_results_position_known_count",
    qualifying_results_summary."qualifying_results_position_mean" AS "qualifying_results_position_mean",
    coalesce(standings_grouped."standings", []) AS "standings",
    coalesce(standings_summary."standings_count", 0) AS "standings_count",
    coalesce(standings_summary."standings_count_30d", 0) AS "standings_count_30d",
    coalesce(standings_summary."standings_count_90d", 0) AS "standings_count_90d",
    coalesce(standings_summary."standings_count_365d", 0) AS "standings_count_365d",
    standings_summary."standings_last_age_days" AS "standings_last_age_days",
    coalesce(standings_summary."standings_distinct_race_id", 0) AS "standings_distinct_race_id",
    coalesce(standings_summary."standings_position_known_count", 0) AS "standings_position_known_count",
    standings_summary."standings_position_mean" AS "standings_position_mean",
    constructor_summary."latest_constructor_id" AS "latest_constructor_id",
    constructor_summary."constructor_results_count" AS "constructor_results_count",
    constructor_summary."constructor_position_known_count" AS "constructor_position_known_count",
    constructor_summary."constructor_position_mean" AS "constructor_position_mean",
    constructor_summary."constructor_points_known_count" AS "constructor_points_known_count",
    constructor_summary."constructor_points_sum" AS "constructor_points_sum"
FROM root_features
LEFT JOIN results_grouped USING ("__row_position")
LEFT JOIN results_summary USING ("__row_position")
LEFT JOIN qualifying_results_grouped USING ("__row_position")
LEFT JOIN qualifying_results_summary USING ("__row_position")
LEFT JOIN standings_grouped USING ("__row_position")
LEFT JOIN standings_summary USING ("__row_position")
LEFT JOIN constructor_summary USING ("__row_position")
LEFT JOIN constructor_race_summary USING ("__row_position")
ORDER BY "__row_position";
