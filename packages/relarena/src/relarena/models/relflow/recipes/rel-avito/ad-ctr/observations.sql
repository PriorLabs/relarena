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
        root."AdID" AS "entity_id",
        root."CategoryID" AS "category_id",
        root."LocationID" AS "location_id",
        root."Title" AS "title"
    FROM anchors AS anchor
    LEFT JOIN "AdsInfo" AS root
        ON anchor."__entity_key" = root."AdID"
    QUALIFY row_number() OVER (
        PARTITION BY anchor."__row_position"
        ORDER BY root."__source_row_id" DESC NULLS LAST
    ) = 1
),
"impressions_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."SearchDate" AS "__event_time",
        event."SearchID" AS "search_id",
        event."IsClick" AS "clicked",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."SearchDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "SearchStream" AS event
        ON anchor."__entity_key" = event."AdID"
        AND event."SearchDate" <= anchor."__as_of"
),
"impressions_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "search_id" := "search_id",
                "clicked" := "clicked"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "impressions"
    FROM "impressions_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"visits_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."ViewDate" AS "__event_time",
        event."UserID" AS "user_id",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."ViewDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "VisitStream" AS event
        ON anchor."__entity_key" = event."AdID"
        AND event."ViewDate" <= anchor."__as_of"
),
"visits_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "user_id" := "user_id"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "visits"
    FROM "visits_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"phone_requests_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."PhoneRequestDate" AS "__event_time",
        event."UserID" AS "user_id",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."PhoneRequestDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "PhoneRequestsStream" AS event
        ON anchor."__entity_key" = event."AdID"
        AND event."PhoneRequestDate" <= anchor."__as_of"
),
"phone_requests_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "user_id" := "user_id"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "phone_requests"
    FROM "phone_requests_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("impressions_grouped"."impressions", []) AS "impressions",
    coalesce("visits_grouped"."visits", []) AS "visits",
    coalesce("phone_requests_grouped"."phone_requests", []) AS "phone_requests"
FROM root_features
LEFT JOIN "impressions_grouped" USING ("__row_position")
LEFT JOIN "visits_grouped" USING ("__row_position")
LEFT JOIN "phone_requests_grouped" USING ("__row_position")
ORDER BY "__row_position";
