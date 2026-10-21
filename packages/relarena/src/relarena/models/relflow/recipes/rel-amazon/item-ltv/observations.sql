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
        root."product_id" AS "entity_id",
        root."category" AS "category",
        root."brand" AS "brand",
        root."title" AS "title",
        root."description" AS "description",
        root."price" AS "price"
    FROM anchors AS anchor
    LEFT JOIN "product" AS root
        ON anchor."__entity_key" = root."product_id"
    QUALIFY row_number() OVER (
        PARTITION BY anchor."__row_position"
        ORDER BY root."__source_row_id" DESC NULLS LAST
    ) = 1
),
"reviews_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."review_time" AS "__event_time",
        event."customer_id" AS "customer_id",
        event."rating" AS "rating",
        event."verified" AS "verified",
        event."summary" AS "summary",
        event."review_text" AS "review_text",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."review_time" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "review" AS event
        ON anchor."__entity_key" = event."product_id"
        AND event."review_time" <= anchor."__as_of"
),
"reviews_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "customer_id" := "customer_id",
                "rating" := "rating",
                "verified" := "verified",
                "summary" := "summary",
                "review_text" := "review_text"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "reviews"
    FROM "reviews_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("reviews_grouped"."reviews", []) AS "reviews"
FROM root_features
LEFT JOIN "reviews_grouped" USING ("__row_position")
ORDER BY "__row_position";
