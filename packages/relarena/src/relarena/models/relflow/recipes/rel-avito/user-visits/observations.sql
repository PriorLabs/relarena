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
        root."UserID" AS "entity_id"
    FROM anchors AS anchor
    LEFT JOIN "UserInfo" AS root
        ON anchor."__entity_key" = root."UserID"
    QUALIFY row_number() OVER (
        PARTITION BY anchor."__row_position"
        ORDER BY root."__source_row_id" DESC NULLS LAST
    ) = 1
),
"searches_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."SearchDate" AS "__event_time",
        event."SearchID" AS "search_id",
        event."CategoryID" AS "category_id",
        event."LocationID" AS "location_id",
        event."SearchQuery" AS "search_query",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."SearchDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "SearchInfo" AS event
        ON anchor."__entity_key" = event."UserID"
        AND event."SearchDate" <= anchor."__as_of"
),
"searches_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "search_id" := "search_id",
                "category_id" := "category_id",
                "location_id" := "location_id",
                "search_query" := "search_query"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "searches"
    FROM "searches_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"visits_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."ViewDate" AS "__event_time",
        event."AdID" AS "ad_id",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."ViewDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "VisitStream" AS event
        ON anchor."__entity_key" = event."UserID"
        AND event."ViewDate" <= anchor."__as_of"
),
"visits_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "ad_id" := "ad_id"
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
        event."AdID" AS "ad_id",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."PhoneRequestDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "PhoneRequestsStream" AS event
        ON anchor."__entity_key" = event."UserID"
        AND event."PhoneRequestDate" <= anchor."__as_of"
),
"phone_requests_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "ad_id" := "ad_id"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "phone_requests"
    FROM "phone_requests_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("searches_grouped"."searches", []) AS "searches",
    coalesce("visits_grouped"."visits", []) AS "visits",
    coalesce("phone_requests_grouped"."phone_requests", []) AS "phone_requests"
FROM root_features
LEFT JOIN "searches_grouped" USING ("__row_position")
LEFT JOIN "visits_grouped" USING ("__row_position")
LEFT JOIN "phone_requests_grouped" USING ("__row_position")
ORDER BY "__row_position";
