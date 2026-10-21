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
        root."article_id" AS "entity_id",
        root."product_code" AS "product_code",
        root."prod_name" AS "product_name",
        root."product_type_no" AS "product_type",
        root."product_group_name" AS "product_group",
        root."graphical_appearance_no" AS "graphical_appearance",
        root."colour_group_code" AS "colour_group",
        root."perceived_colour_value_id" AS "perceived_colour_value",
        root."perceived_colour_master_id" AS "perceived_colour_master",
        root."department_no" AS "department",
        root."index_code" AS "index_code",
        root."index_group_no" AS "index_group",
        root."section_no" AS "section",
        root."garment_group_no" AS "garment_group",
        root."detail_desc" AS "detail_description"
    FROM anchors AS anchor
    LEFT JOIN "article" AS root
        ON anchor."__entity_key" = root."article_id"
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
        event."customer_id" AS "customer_id",
        event."price" AS "price",
        event."sales_channel_id" AS "sales_channel",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."t_dat" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "transactions" AS event
        ON anchor."__entity_key" = event."article_id"
        AND event."t_dat" <= anchor."__as_of"
),
"transactions_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "customer_id" := "customer_id",
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
