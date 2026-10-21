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
        root."customer_id" AS "entity_id",
        root."FN" AS "fn",
        root."Active" AS "active",
        root."club_member_status" AS "club_member_status",
        root."fashion_news_frequency" AS "fashion_news_frequency",
        root."age" AS "age",
        root."postal_code" AS "postal_code"
    FROM anchors AS anchor
    LEFT JOIN "customer" AS root
        ON anchor."__entity_key" = root."customer_id"
    QUALIFY row_number() OVER (
        PARTITION BY anchor."__row_position"
        ORDER BY root."__source_row_id" DESC NULLS LAST
    ) = 1
),
"transactions_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."t_dat" AS "__event_time",
        event."article_id" AS "article_id",
        event."price" AS "price",
        event."sales_channel_id" AS "sales_channel",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."t_dat" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "transactions" AS event
        ON anchor."__entity_key" = event."customer_id"
        AND event."t_dat" <= anchor."__as_of"
),
"transactions_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "article_id" := "article_id",
                "price" := "price",
                "sales_channel" := "sales_channel"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "transactions"
    FROM "transactions_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("transactions_grouped"."transactions", []) AS "transactions"
FROM root_features
LEFT JOIN "transactions_grouped" USING ("__row_position")
ORDER BY "__row_position";
