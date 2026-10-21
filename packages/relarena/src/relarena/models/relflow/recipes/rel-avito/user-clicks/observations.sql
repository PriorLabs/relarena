-- Rolling windows are (as_of - duration, as_of]; unknown rates and recencies remain null.
-- Feature-only query; all event summaries precede the independent latest-64 caps.
-- HistCTR is the recorded historical estimate on each cutoff-eligible impression, never a recomputed future rate.
-- Static descriptive tables follow RelArena's database contract; no mutable snapshot counters are used.
WITH
anchors AS (
    SELECT "__row_position", "__entity_key", "__as_of"
    FROM read_parquet($anchors)
    WHERE "__row_position" >= $start AND "__row_position" < $stop
),
"ads" AS (
    SELECT * FROM "AdsInfo"
    QUALIFY row_number() OVER (PARTITION BY "AdID" ORDER BY "__source_row_id" DESC) = 1
),
"category_context" AS (
    SELECT * FROM "Category"
    QUALIFY row_number() OVER (PARTITION BY "CategoryID" ORDER BY "__source_row_id" DESC) = 1
),
"location_context" AS (
    SELECT * FROM "Location"
    QUALIFY row_number() OVER (PARTITION BY "LocationID" ORDER BY "__source_row_id" DESC) = 1
),
"user_context" AS (
    SELECT * FROM "UserInfo"
    QUALIFY row_number() OVER (PARTITION BY "UserID" ORDER BY "__source_row_id" DESC) = 1
),
"root_features" AS (
    SELECT
        anchor."__row_position" AS "__row_position",
        anchor."__as_of" AS "as_of",
        epoch(anchor."__as_of") / 86400.0 AS "as_of_days",
        CASE WHEN root."UserID" IS NULL THEN NULL ELSE 'rel-avito:user:' || CAST(try_cast(root."UserID" AS BIGINT) AS VARCHAR) END AS "entity_id",
        CASE WHEN root."UserAgentID" IS NOT NULL THEN 'rel-avito:user-agent:' || CAST(try_cast(root."UserAgentID" AS BIGINT) AS VARCHAR) END AS "user_agent_id",
        root."UserAgentOSID" AS "user_agent_os_id",
        root."UserDeviceID" AS "user_device_id",
        root."UserAgentFamilyID" AS "user_agent_family_id"
    FROM anchors AS anchor
    LEFT JOIN "UserInfo" AS root ON anchor."__entity_key" = root."UserID"
    QUALIFY row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY root."__source_row_id" DESC NULLS LAST) = 1
),
"searches_events" AS (
    SELECT
        anchor."__row_position" AS "__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."SearchDate" AS "__event_time",
        date_diff('microseconds', event."SearchDate", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."SearchID" IS NULL THEN NULL ELSE 'rel-avito:search:' || CAST(try_cast(event."SearchID" AS BIGINT) AS VARCHAR) END AS "search_id",
        event."CategoryID" AS "category_id",
        event."LocationID" AS "location_id",
        event."SearchQuery" AS "search_query",
        category."Level" AS "category_level",
        category."ParentCategoryID" AS "parent_category_id",
        category."SubcategoryID" AS "subcategory_id",
        location."Level" AS "location_level",
        location."RegionID" AS "region_id",
        location."CityID" AS "city_id",
        CASE WHEN try_cast(event."IsUserLoggedOn" AS DOUBLE) IN (0, 1) THEN try_cast(event."IsUserLoggedOn" AS DOUBLE) = 1 ELSE NULL END AS "is_user_logged_on",
        CASE WHEN event."IPID" IS NOT NULL THEN 'rel-avito:ip:' || CAST(try_cast(event."IPID" AS BIGINT) AS VARCHAR) END AS "ip_id"
    FROM anchors AS anchor
    INNER JOIN "SearchInfo" AS event
        ON anchor."__entity_key" = event."UserID"
        AND event."SearchDate" <= anchor."__as_of"
    LEFT JOIN category_context AS category ON category."CategoryID" = event."CategoryID"
    LEFT JOIN location_context AS location ON location."LocationID" = event."LocationID"
),
"searches_ranked" AS (
    SELECT *, row_number() OVER (
        PARTITION BY "__row_position" ORDER BY "__event_time" DESC, "__source_order" DESC
    ) AS "__recency"
    FROM "searches_events"
),
"searches_grouped" AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "search_id" := "search_id",
                "category_id" := "category_id",
                "location_id" := "location_id",
                "search_query" := "search_query",
                "category_level" := "category_level",
                "parent_category_id" := "parent_category_id",
                "subcategory_id" := "subcategory_id",
                "location_level" := "location_level",
                "region_id" := "region_id",
                "city_id" := "city_id",
                "is_user_logged_on" := "is_user_logged_on",
                "ip_id" := "ip_id"
            ) ORDER BY "__event_time" ASC, "__source_order" ASC) AS "searches"
    FROM "searches_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"searches_summary" AS (
    SELECT "__row_position",
        count(*) AS "searches_count",
        count(*) FILTER (WHERE "age_days" < 4) AS "searches_count_4d",
        count(*) FILTER (WHERE "age_days" < 7) AS "searches_count_7d",
        count(*) FILTER (WHERE "age_days" < 30) AS "searches_count_30d",
        count(*) FILTER (WHERE "age_days" < 4) - count(*) FILTER (WHERE "age_days" >= 4 AND "age_days" < 8) AS "searches_recent_change",
        count(DISTINCT CAST("__event_time" AS DATE)) FILTER (WHERE "age_days" < 30) AS "searches_active_days_30d",
        count(DISTINCT "category_id") AS "searches_distinct_categories",
        min("age_days") AS "searches_last_age_days",
        max("age_days") AS "searches_first_age_days"
    FROM "searches_events"
    GROUP BY "__row_position"
),
"impressions_events" AS (
    SELECT
        anchor."__row_position" AS "__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."SearchDate" AS "__event_time",
        date_diff('microseconds', event."SearchDate", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."SearchID" IS NULL THEN NULL ELSE 'rel-avito:search:' || CAST(try_cast(event."SearchID" AS BIGINT) AS VARCHAR) END AS "search_id",
        CASE WHEN try_cast(event."IsClick" AS DOUBLE) IN (0, 1) THEN try_cast(event."IsClick" AS DOUBLE) = 1 ELSE NULL END AS "clicked",
        context."CategoryID" AS "category_id",
        context."LocationID" AS "location_id",
        context."SearchQuery" AS "search_query",
        CASE WHEN event."AdID" IS NULL THEN NULL ELSE 'rel-avito:ad:' || CAST(try_cast(event."AdID" AS BIGINT) AS VARCHAR) END AS "ad_id",
        ad."CategoryID" AS "ad_category_id",
        ad."LocationID" AS "ad_location_id",
        ad."Title" AS "ad_title",
        ad."Price" AS "ad_price",
        CASE WHEN try_cast(ad."IsContext" AS DOUBLE) IN (0, 1) THEN try_cast(ad."IsContext" AS DOUBLE) = 1 ELSE NULL END AS "ad_is_context",
        ad_category."Level" AS "ad_category_level",
        ad_category."ParentCategoryID" AS "ad_parent_category_id",
        ad_category."SubcategoryID" AS "ad_subcategory_id",
        ad_location."Level" AS "ad_location_level",
        ad_location."RegionID" AS "ad_region_id",
        ad_location."CityID" AS "ad_city_id",
        category."Level" AS "category_level",
        category."ParentCategoryID" AS "parent_category_id",
        category."SubcategoryID" AS "subcategory_id",
        location."Level" AS "location_level",
        location."RegionID" AS "region_id",
        location."CityID" AS "city_id",
        event."Position" AS "position",
        event."ObjectType" AS "object_type",
        event."HistCTR" AS "historical_ctr",
        CASE WHEN try_cast(context."IsUserLoggedOn" AS DOUBLE) IN (0, 1) THEN try_cast(context."IsUserLoggedOn" AS DOUBLE) = 1 ELSE NULL END AS "is_user_logged_on",
        CASE WHEN context."IPID" IS NOT NULL THEN 'rel-avito:ip:' || CAST(try_cast(context."IPID" AS BIGINT) AS VARCHAR) END AS "ip_id"
    FROM anchors AS anchor
    INNER JOIN "SearchInfo" AS context
        ON context."UserID" = anchor."__entity_key" AND context."SearchDate" <= anchor."__as_of"
    INNER JOIN "SearchStream" AS event
        ON event."SearchID" = context."SearchID" AND event."SearchDate" <= anchor."__as_of"
    LEFT JOIN ads AS ad ON ad."AdID" = event."AdID"
    LEFT JOIN category_context AS ad_category ON ad_category."CategoryID" = ad."CategoryID"
    LEFT JOIN location_context AS ad_location ON ad_location."LocationID" = ad."LocationID"
    LEFT JOIN category_context AS category ON category."CategoryID" = context."CategoryID"
    LEFT JOIN location_context AS location ON location."LocationID" = context."LocationID"
    QUALIFY row_number() OVER (PARTITION BY anchor."__row_position", event."__source_row_id" ORDER BY context."SearchDate" DESC NULLS LAST, context."__source_row_id" DESC NULLS LAST) = 1
),
"impressions_ranked" AS (
    SELECT *, row_number() OVER (
        PARTITION BY "__row_position" ORDER BY "__event_time" DESC, "__source_order" DESC
    ) AS "__recency"
    FROM "impressions_events"
),
"impressions_grouped" AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "search_id" := "search_id",
                "ad_id" := "ad_id",
                "clicked" := "clicked",
                "category_id" := "category_id",
                "location_id" := "location_id",
                "search_query" := "search_query",
                "ad_category_id" := "ad_category_id",
                "ad_location_id" := "ad_location_id",
                "ad_title" := "ad_title",
                "ad_price" := "ad_price",
                "ad_is_context" := "ad_is_context",
                "ad_category_level" := "ad_category_level",
                "ad_parent_category_id" := "ad_parent_category_id",
                "ad_subcategory_id" := "ad_subcategory_id",
                "ad_location_level" := "ad_location_level",
                "ad_region_id" := "ad_region_id",
                "ad_city_id" := "ad_city_id",
                "category_level" := "category_level",
                "parent_category_id" := "parent_category_id",
                "subcategory_id" := "subcategory_id",
                "location_level" := "location_level",
                "region_id" := "region_id",
                "city_id" := "city_id",
                "position" := "position",
                "object_type" := "object_type",
                "historical_ctr" := "historical_ctr",
                "is_user_logged_on" := "is_user_logged_on",
                "ip_id" := "ip_id"
            ) ORDER BY "__event_time" ASC, "__source_order" ASC) AS "impressions"
    FROM "impressions_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"impressions_summary" AS (
    SELECT "__row_position",
        count(*) AS "impressions_count",
        count(*) FILTER (WHERE "age_days" < 4) AS "impressions_count_4d",
        count(*) FILTER (WHERE "age_days" < 7) AS "impressions_count_7d",
        count(*) FILTER (WHERE "age_days" < 30) AS "impressions_count_30d",
        count(*) FILTER (WHERE "age_days" < 4) - count(*) FILTER (WHERE "age_days" >= 4 AND "age_days" < 8) AS "impressions_recent_change",
        count(DISTINCT CAST("__event_time" AS DATE)) FILTER (WHERE "age_days" < 30) AS "impressions_active_days_30d",
        count(DISTINCT "ad_id") AS "impressions_distinct_ads",
        count(*) FILTER (WHERE "clicked") AS "impressions_click_count",
        count("clicked") AS "impressions_known_click_count",
        min("age_days") AS "impressions_last_age_days",
        max("age_days") AS "impressions_first_age_days",
        count(*) FILTER (WHERE "clicked")::DOUBLE / nullif(count("clicked"), 0) AS "impressions_ctr",
        CASE WHEN count("clicked") > 0 THEN (count(*) FILTER (WHERE "clicked") + 1.0) / (count("clicked") + 2.0) END AS "impressions_smoothed_ctr",
        count(*) FILTER (WHERE "clicked" AND "age_days" < 4)::DOUBLE / nullif(count("clicked") FILTER (WHERE "age_days" < 4), 0) AS "impressions_ctr_4d",
        count(*) FILTER (WHERE "clicked" AND "age_days" < 30)::DOUBLE / nullif(count("clicked") FILTER (WHERE "age_days" < 30), 0) AS "impressions_ctr_30d"
    FROM "impressions_events"
    GROUP BY "__row_position"
),
"visits_events" AS (
    SELECT
        anchor."__row_position" AS "__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."ViewDate" AS "__event_time",
        date_diff('microseconds', event."ViewDate", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."AdID" IS NULL THEN NULL ELSE 'rel-avito:ad:' || CAST(try_cast(event."AdID" AS BIGINT) AS VARCHAR) END AS "ad_id",
        ad."CategoryID" AS "ad_category_id",
        ad."LocationID" AS "ad_location_id",
        ad."Title" AS "ad_title",
        ad."Price" AS "ad_price",
        CASE WHEN try_cast(ad."IsContext" AS DOUBLE) IN (0, 1) THEN try_cast(ad."IsContext" AS DOUBLE) = 1 ELSE NULL END AS "ad_is_context",
        ad_category."Level" AS "ad_category_level",
        ad_category."ParentCategoryID" AS "ad_parent_category_id",
        ad_category."SubcategoryID" AS "ad_subcategory_id",
        ad_location."Level" AS "ad_location_level",
        ad_location."RegionID" AS "ad_region_id",
        ad_location."CityID" AS "ad_city_id",
        CASE WHEN event."IPID" IS NOT NULL THEN 'rel-avito:ip:' || CAST(try_cast(event."IPID" AS BIGINT) AS VARCHAR) END AS "ip_id"
    FROM anchors AS anchor
    INNER JOIN "VisitStream" AS event
        ON anchor."__entity_key" = event."UserID"
        AND event."ViewDate" <= anchor."__as_of"
    LEFT JOIN ads AS ad ON ad."AdID" = event."AdID"
    LEFT JOIN category_context AS ad_category ON ad_category."CategoryID" = ad."CategoryID"
    LEFT JOIN location_context AS ad_location ON ad_location."LocationID" = ad."LocationID"
),
"visits_ranked" AS (
    SELECT *, row_number() OVER (
        PARTITION BY "__row_position" ORDER BY "__event_time" DESC, "__source_order" DESC
    ) AS "__recency"
    FROM "visits_events"
),
"visits_grouped" AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "ad_id" := "ad_id",
                "ad_category_id" := "ad_category_id",
                "ad_location_id" := "ad_location_id",
                "ad_title" := "ad_title",
                "ad_price" := "ad_price",
                "ad_is_context" := "ad_is_context",
                "ad_category_level" := "ad_category_level",
                "ad_parent_category_id" := "ad_parent_category_id",
                "ad_subcategory_id" := "ad_subcategory_id",
                "ad_location_level" := "ad_location_level",
                "ad_region_id" := "ad_region_id",
                "ad_city_id" := "ad_city_id",
                "ip_id" := "ip_id"
            ) ORDER BY "__event_time" ASC, "__source_order" ASC) AS "visits"
    FROM "visits_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"visits_summary" AS (
    SELECT "__row_position",
        count(*) AS "visits_count",
        count(*) FILTER (WHERE "age_days" < 4) AS "visits_count_4d",
        count(*) FILTER (WHERE "age_days" < 7) AS "visits_count_7d",
        count(*) FILTER (WHERE "age_days" < 30) AS "visits_count_30d",
        count(*) FILTER (WHERE "age_days" < 4) - count(*) FILTER (WHERE "age_days" >= 4 AND "age_days" < 8) AS "visits_recent_change",
        count(DISTINCT CAST("__event_time" AS DATE)) FILTER (WHERE "age_days" < 30) AS "visits_active_days_30d",
        count(DISTINCT "ad_id") AS "visits_distinct_ads",
        min("age_days") AS "visits_last_age_days",
        max("age_days") AS "visits_first_age_days"
    FROM "visits_events"
    GROUP BY "__row_position"
),
"phone_requests_events" AS (
    SELECT
        anchor."__row_position" AS "__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."PhoneRequestDate" AS "__event_time",
        date_diff('microseconds', event."PhoneRequestDate", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."AdID" IS NULL THEN NULL ELSE 'rel-avito:ad:' || CAST(try_cast(event."AdID" AS BIGINT) AS VARCHAR) END AS "ad_id",
        ad."CategoryID" AS "ad_category_id",
        ad."LocationID" AS "ad_location_id",
        ad."Title" AS "ad_title",
        ad."Price" AS "ad_price",
        CASE WHEN try_cast(ad."IsContext" AS DOUBLE) IN (0, 1) THEN try_cast(ad."IsContext" AS DOUBLE) = 1 ELSE NULL END AS "ad_is_context",
        ad_category."Level" AS "ad_category_level",
        ad_category."ParentCategoryID" AS "ad_parent_category_id",
        ad_category."SubcategoryID" AS "ad_subcategory_id",
        ad_location."Level" AS "ad_location_level",
        ad_location."RegionID" AS "ad_region_id",
        ad_location."CityID" AS "ad_city_id",
        CASE WHEN event."IPID" IS NOT NULL THEN 'rel-avito:ip:' || CAST(try_cast(event."IPID" AS BIGINT) AS VARCHAR) END AS "ip_id"
    FROM anchors AS anchor
    INNER JOIN "PhoneRequestsStream" AS event
        ON anchor."__entity_key" = event."UserID"
        AND event."PhoneRequestDate" <= anchor."__as_of"
    LEFT JOIN ads AS ad ON ad."AdID" = event."AdID"
    LEFT JOIN category_context AS ad_category ON ad_category."CategoryID" = ad."CategoryID"
    LEFT JOIN location_context AS ad_location ON ad_location."LocationID" = ad."LocationID"
),
"phone_requests_ranked" AS (
    SELECT *, row_number() OVER (
        PARTITION BY "__row_position" ORDER BY "__event_time" DESC, "__source_order" DESC
    ) AS "__recency"
    FROM "phone_requests_events"
),
"phone_requests_grouped" AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "ad_id" := "ad_id",
                "ad_category_id" := "ad_category_id",
                "ad_location_id" := "ad_location_id",
                "ad_title" := "ad_title",
                "ad_price" := "ad_price",
                "ad_is_context" := "ad_is_context",
                "ad_category_level" := "ad_category_level",
                "ad_parent_category_id" := "ad_parent_category_id",
                "ad_subcategory_id" := "ad_subcategory_id",
                "ad_location_level" := "ad_location_level",
                "ad_region_id" := "ad_region_id",
                "ad_city_id" := "ad_city_id",
                "ip_id" := "ip_id"
            ) ORDER BY "__event_time" ASC, "__source_order" ASC) AS "phone_requests"
    FROM "phone_requests_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"phone_requests_summary" AS (
    SELECT "__row_position",
        count(*) AS "phone_requests_count",
        count(*) FILTER (WHERE "age_days" < 4) AS "phone_requests_count_4d",
        count(*) FILTER (WHERE "age_days" < 7) AS "phone_requests_count_7d",
        count(*) FILTER (WHERE "age_days" < 30) AS "phone_requests_count_30d",
        count(*) FILTER (WHERE "age_days" < 4) - count(*) FILTER (WHERE "age_days" >= 4 AND "age_days" < 8) AS "phone_requests_recent_change",
        count(DISTINCT CAST("__event_time" AS DATE)) FILTER (WHERE "age_days" < 30) AS "phone_requests_active_days_30d",
        count(DISTINCT "ad_id") AS "phone_requests_distinct_ads",
        min("age_days") AS "phone_requests_last_age_days",
        max("age_days") AS "phone_requests_first_age_days"
    FROM "phone_requests_events"
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("searches_grouped"."searches", []) AS "searches",
    coalesce("searches_summary"."searches_count", 0) AS "searches_count",
    coalesce("searches_summary"."searches_count_4d", 0) AS "searches_count_4d",
    coalesce("searches_summary"."searches_count_7d", 0) AS "searches_count_7d",
    coalesce("searches_summary"."searches_count_30d", 0) AS "searches_count_30d",
    coalesce("searches_summary"."searches_recent_change", 0) AS "searches_recent_change",
    coalesce("searches_summary"."searches_active_days_30d", 0) AS "searches_active_days_30d",
    coalesce("searches_summary"."searches_distinct_categories", 0) AS "searches_distinct_categories",
    "searches_summary"."searches_last_age_days",
    "searches_summary"."searches_first_age_days",
    coalesce("impressions_grouped"."impressions", []) AS "impressions",
    coalesce("impressions_summary"."impressions_count", 0) AS "impressions_count",
    coalesce("impressions_summary"."impressions_count_4d", 0) AS "impressions_count_4d",
    coalesce("impressions_summary"."impressions_count_7d", 0) AS "impressions_count_7d",
    coalesce("impressions_summary"."impressions_count_30d", 0) AS "impressions_count_30d",
    coalesce("impressions_summary"."impressions_recent_change", 0) AS "impressions_recent_change",
    coalesce("impressions_summary"."impressions_active_days_30d", 0) AS "impressions_active_days_30d",
    coalesce("impressions_summary"."impressions_distinct_ads", 0) AS "impressions_distinct_ads",
    coalesce("impressions_summary"."impressions_click_count", 0) AS "impressions_click_count",
    coalesce("impressions_summary"."impressions_known_click_count", 0) AS "impressions_known_click_count",
    "impressions_summary"."impressions_last_age_days",
    "impressions_summary"."impressions_first_age_days",
    "impressions_summary"."impressions_ctr",
    "impressions_summary"."impressions_smoothed_ctr",
    "impressions_summary"."impressions_ctr_4d",
    "impressions_summary"."impressions_ctr_30d",
    coalesce("visits_grouped"."visits", []) AS "visits",
    coalesce("visits_summary"."visits_count", 0) AS "visits_count",
    coalesce("visits_summary"."visits_count_4d", 0) AS "visits_count_4d",
    coalesce("visits_summary"."visits_count_7d", 0) AS "visits_count_7d",
    coalesce("visits_summary"."visits_count_30d", 0) AS "visits_count_30d",
    coalesce("visits_summary"."visits_recent_change", 0) AS "visits_recent_change",
    coalesce("visits_summary"."visits_active_days_30d", 0) AS "visits_active_days_30d",
    coalesce("visits_summary"."visits_distinct_ads", 0) AS "visits_distinct_ads",
    "visits_summary"."visits_last_age_days",
    "visits_summary"."visits_first_age_days",
    coalesce("phone_requests_grouped"."phone_requests", []) AS "phone_requests",
    coalesce("phone_requests_summary"."phone_requests_count", 0) AS "phone_requests_count",
    coalesce("phone_requests_summary"."phone_requests_count_4d", 0) AS "phone_requests_count_4d",
    coalesce("phone_requests_summary"."phone_requests_count_7d", 0) AS "phone_requests_count_7d",
    coalesce("phone_requests_summary"."phone_requests_count_30d", 0) AS "phone_requests_count_30d",
    coalesce("phone_requests_summary"."phone_requests_recent_change", 0) AS "phone_requests_recent_change",
    coalesce("phone_requests_summary"."phone_requests_active_days_30d", 0) AS "phone_requests_active_days_30d",
    coalesce("phone_requests_summary"."phone_requests_distinct_ads", 0) AS "phone_requests_distinct_ads",
    "phone_requests_summary"."phone_requests_last_age_days",
    "phone_requests_summary"."phone_requests_first_age_days",
    coalesce("phone_requests_summary"."phone_requests_count", 0)::DOUBLE / nullif("visits_summary"."visits_count", 0) AS "phone_to_visit_ratio"
FROM root_features
LEFT JOIN "searches_grouped" USING ("__row_position")
LEFT JOIN "searches_summary" USING ("__row_position")
LEFT JOIN "impressions_grouped" USING ("__row_position")
LEFT JOIN "impressions_summary" USING ("__row_position")
LEFT JOIN "visits_grouped" USING ("__row_position")
LEFT JOIN "visits_summary" USING ("__row_position")
LEFT JOIN "phone_requests_grouped" USING ("__row_position")
LEFT JOIN "phone_requests_summary" USING ("__row_position")
ORDER BY "__row_position";
