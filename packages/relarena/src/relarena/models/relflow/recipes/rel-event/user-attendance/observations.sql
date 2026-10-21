-- Labels remain in runner-owned anchors. Timestamped rows obey the scoring cutoff.
-- Static attributes follow the benchmark snapshot; historical attribute versions are not supplied.
-- Content coordinates retain c_1..c_100,c_other order; incomplete vectors stay null.
-- count_if outputs are BIGINT to retain Number-compatible Arrow integer atoms.
-- Raw history caps do not restrict the full-history summaries.
WITH
anchors AS (
    SELECT "__row_position", "__entity_key", "__as_of"
    FROM read_parquet($anchors)
    WHERE "__row_position" >= $start AND "__row_position" < $stop
),
event_values AS (
    SELECT content.*,
        [
            TRY_CAST(content."c_1" AS DOUBLE), TRY_CAST(content."c_2" AS DOUBLE), TRY_CAST(content."c_3" AS DOUBLE),
            TRY_CAST(content."c_4" AS DOUBLE), TRY_CAST(content."c_5" AS DOUBLE), TRY_CAST(content."c_6" AS DOUBLE),
            TRY_CAST(content."c_7" AS DOUBLE), TRY_CAST(content."c_8" AS DOUBLE), TRY_CAST(content."c_9" AS DOUBLE),
            TRY_CAST(content."c_10" AS DOUBLE), TRY_CAST(content."c_11" AS DOUBLE), TRY_CAST(content."c_12" AS DOUBLE),
            TRY_CAST(content."c_13" AS DOUBLE), TRY_CAST(content."c_14" AS DOUBLE), TRY_CAST(content."c_15" AS DOUBLE),
            TRY_CAST(content."c_16" AS DOUBLE), TRY_CAST(content."c_17" AS DOUBLE), TRY_CAST(content."c_18" AS DOUBLE),
            TRY_CAST(content."c_19" AS DOUBLE), TRY_CAST(content."c_20" AS DOUBLE), TRY_CAST(content."c_21" AS DOUBLE),
            TRY_CAST(content."c_22" AS DOUBLE), TRY_CAST(content."c_23" AS DOUBLE), TRY_CAST(content."c_24" AS DOUBLE),
            TRY_CAST(content."c_25" AS DOUBLE), TRY_CAST(content."c_26" AS DOUBLE), TRY_CAST(content."c_27" AS DOUBLE),
            TRY_CAST(content."c_28" AS DOUBLE), TRY_CAST(content."c_29" AS DOUBLE), TRY_CAST(content."c_30" AS DOUBLE),
            TRY_CAST(content."c_31" AS DOUBLE), TRY_CAST(content."c_32" AS DOUBLE), TRY_CAST(content."c_33" AS DOUBLE),
            TRY_CAST(content."c_34" AS DOUBLE), TRY_CAST(content."c_35" AS DOUBLE), TRY_CAST(content."c_36" AS DOUBLE),
            TRY_CAST(content."c_37" AS DOUBLE), TRY_CAST(content."c_38" AS DOUBLE), TRY_CAST(content."c_39" AS DOUBLE),
            TRY_CAST(content."c_40" AS DOUBLE), TRY_CAST(content."c_41" AS DOUBLE), TRY_CAST(content."c_42" AS DOUBLE),
            TRY_CAST(content."c_43" AS DOUBLE), TRY_CAST(content."c_44" AS DOUBLE), TRY_CAST(content."c_45" AS DOUBLE),
            TRY_CAST(content."c_46" AS DOUBLE), TRY_CAST(content."c_47" AS DOUBLE), TRY_CAST(content."c_48" AS DOUBLE),
            TRY_CAST(content."c_49" AS DOUBLE), TRY_CAST(content."c_50" AS DOUBLE), TRY_CAST(content."c_51" AS DOUBLE),
            TRY_CAST(content."c_52" AS DOUBLE), TRY_CAST(content."c_53" AS DOUBLE), TRY_CAST(content."c_54" AS DOUBLE),
            TRY_CAST(content."c_55" AS DOUBLE), TRY_CAST(content."c_56" AS DOUBLE), TRY_CAST(content."c_57" AS DOUBLE),
            TRY_CAST(content."c_58" AS DOUBLE), TRY_CAST(content."c_59" AS DOUBLE), TRY_CAST(content."c_60" AS DOUBLE),
            TRY_CAST(content."c_61" AS DOUBLE), TRY_CAST(content."c_62" AS DOUBLE), TRY_CAST(content."c_63" AS DOUBLE),
            TRY_CAST(content."c_64" AS DOUBLE), TRY_CAST(content."c_65" AS DOUBLE), TRY_CAST(content."c_66" AS DOUBLE),
            TRY_CAST(content."c_67" AS DOUBLE), TRY_CAST(content."c_68" AS DOUBLE), TRY_CAST(content."c_69" AS DOUBLE),
            TRY_CAST(content."c_70" AS DOUBLE), TRY_CAST(content."c_71" AS DOUBLE), TRY_CAST(content."c_72" AS DOUBLE),
            TRY_CAST(content."c_73" AS DOUBLE), TRY_CAST(content."c_74" AS DOUBLE), TRY_CAST(content."c_75" AS DOUBLE),
            TRY_CAST(content."c_76" AS DOUBLE), TRY_CAST(content."c_77" AS DOUBLE), TRY_CAST(content."c_78" AS DOUBLE),
            TRY_CAST(content."c_79" AS DOUBLE), TRY_CAST(content."c_80" AS DOUBLE), TRY_CAST(content."c_81" AS DOUBLE),
            TRY_CAST(content."c_82" AS DOUBLE), TRY_CAST(content."c_83" AS DOUBLE), TRY_CAST(content."c_84" AS DOUBLE),
            TRY_CAST(content."c_85" AS DOUBLE), TRY_CAST(content."c_86" AS DOUBLE), TRY_CAST(content."c_87" AS DOUBLE),
            TRY_CAST(content."c_88" AS DOUBLE), TRY_CAST(content."c_89" AS DOUBLE), TRY_CAST(content."c_90" AS DOUBLE),
            TRY_CAST(content."c_91" AS DOUBLE), TRY_CAST(content."c_92" AS DOUBLE), TRY_CAST(content."c_93" AS DOUBLE),
            TRY_CAST(content."c_94" AS DOUBLE), TRY_CAST(content."c_95" AS DOUBLE), TRY_CAST(content."c_96" AS DOUBLE),
            TRY_CAST(content."c_97" AS DOUBLE), TRY_CAST(content."c_98" AS DOUBLE), TRY_CAST(content."c_99" AS DOUBLE),
            TRY_CAST(content."c_100" AS DOUBLE), TRY_CAST(content."c_other" AS DOUBLE)
        ] AS "__content_values"
    FROM "events" AS content
),
event_content AS (
    SELECT * EXCLUDE ("__content_values"),
        CASE WHEN list_count("__content_values") = 101 THEN "__content_values" ELSE NULL END AS "content_counts"
    FROM event_values
),
root_features AS (
    SELECT anchor."__row_position",
        anchor."__as_of" AS "as_of",
        extract(year FROM anchor."__as_of") AS "as_of_year",
        root."user_id" IS NOT NULL AS "entity_available",
        CASE WHEN root."user_id" IS NOT NULL THEN 'rel-event:user:' || CAST(root."user_id" AS VARCHAR) END AS "entity_id",
        root."locale" AS "locale",
        TRY_CAST(root."birthyear" AS DOUBLE) AS "birthyear",
        root."gender" AS "gender",
        root."location" AS "location",
        TRY_CAST(root."timezone" AS DOUBLE) AS "timezone",
        root."joinedAt" AS "joined_at",
        date_diff('microseconds', root."joinedAt", anchor."__as_of") / 86400000000.0 AS "tenure_days",
        extract(year FROM anchor."__as_of") - TRY_CAST(root."birthyear" AS DOUBLE) AS "age_years"
    FROM anchors AS anchor
    LEFT JOIN "users" AS root ON anchor."__entity_key" = root."user_id" AND root."joinedAt" <= anchor."__as_of"
    QUALIFY row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY root."joinedAt" DESC NULLS LAST, root."__source_row_id" DESC NULLS LAST) = 1
),
attendance_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."start_time" AS "__event_time",
        date_diff('microseconds', event."start_time", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."event" IS NOT NULL THEN 'rel-event:event:' || CAST(event."event" AS VARCHAR) END AS "event_id",
        event."start_time" AS "event_start",
        event."status" AS "status",
        CASE WHEN details."user_id" IS NOT NULL THEN 'rel-event:user:' || CAST(details."user_id" AS VARCHAR) END AS "organizer_id",
        details."city" AS "city",
        details."state" AS "state",
        CASE WHEN details."zip" IS NOT NULL THEN 'rel-event:postal:' || CAST(details."zip" AS VARCHAR) END AS "zip_code",
        details."country" AS "country",
        TRY_CAST(details."lat" AS DOUBLE) AS "latitude",
        TRY_CAST(details."lng" AS DOUBLE) AS "longitude",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."start_time" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency",
        details."content_counts" AS "content_counts"
    FROM anchors AS anchor
    INNER JOIN "event_attendees" AS event ON anchor."__entity_key" = event."user_id" AND event."start_time" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM event_content AS linked WHERE linked."event_id" = event."event" AND linked."start_time" <= anchor."__as_of" ORDER BY linked."start_time" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS details ON TRUE
),
attendance_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "event_id" := "event_id",
                "event_start" := "event_start",
                "status" := "status",
                "organizer_id" := "organizer_id",
                "city" := "city",
                "state" := "state",
                "zip_code" := "zip_code",
                "country" := "country",
                "latitude" := "latitude",
                "longitude" := "longitude",
                "content_counts" := "content_counts"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "attendance"
    FROM attendance_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
attendance_summary AS (
    SELECT "__row_position",
        count(*) AS "attendance_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "attendance_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "attendance_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "attendance_count_365d",
        min("age_days") AS "attendance_last_age_days",
        count(DISTINCT "event_id") AS "attendance_distinct_event_id",
        count(DISTINCT "organizer_id") AS "attendance_distinct_organizer_id",
        count("status") AS "attendance_status_known_count",
        CAST(count_if(lower(CAST("status" AS VARCHAR)) IN ('yes', 'maybe')) AS BIGINT) AS "attendance_accepted_count",
        CAST(count_if(lower(CAST("status" AS VARCHAR)) = 'invited') AS BIGINT) AS "attendance_invited_count",
        CAST(count_if(lower(CAST("status" AS VARCHAR)) IN ('yes', 'maybe')) AS BIGINT) / nullif(count("status"), 0) AS "attendance_accepted_rate",
        count("event_id") - count(DISTINCT "event_id") AS "attendance_repeated_event_count"
    FROM attendance_ranked
    GROUP BY "__row_position"
),
interests_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."timestamp" AS "__event_time",
        date_diff('microseconds', event."timestamp", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."event" IS NOT NULL THEN 'rel-event:event:' || CAST(event."event" AS VARCHAR) END AS "event_id",
        event."timestamp" AS "response_time",
        details."start_time" AS "event_start",
        event."invited" AS "invited",
        event."interested" AS "interested",
        event."not_interested" AS "not_interested",
        CASE WHEN details."user_id" IS NOT NULL THEN 'rel-event:user:' || CAST(details."user_id" AS VARCHAR) END AS "organizer_id",
        details."city" AS "city",
        details."state" AS "state",
        CASE WHEN details."zip" IS NOT NULL THEN 'rel-event:postal:' || CAST(details."zip" AS VARCHAR) END AS "zip_code",
        details."country" AS "country",
        TRY_CAST(details."lat" AS DOUBLE) AS "latitude",
        TRY_CAST(details."lng" AS DOUBLE) AS "longitude",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."timestamp" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency",
        details."content_counts" AS "content_counts"
    FROM anchors AS anchor
    INNER JOIN "event_interest" AS event ON anchor."__entity_key" = event."user" AND event."timestamp" <= anchor."__as_of"
    LEFT JOIN LATERAL (SELECT linked.* FROM event_content AS linked WHERE linked."event_id" = event."event" AND linked."start_time" <= anchor."__as_of" ORDER BY linked."start_time" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS details ON TRUE
),
interests_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "event_id" := "event_id",
                "response_time" := "response_time",
                "event_start" := "event_start",
                "invited" := "invited",
                "interested" := "interested",
                "not_interested" := "not_interested",
                "organizer_id" := "organizer_id",
                "city" := "city",
                "state" := "state",
                "zip_code" := "zip_code",
                "country" := "country",
                "latitude" := "latitude",
                "longitude" := "longitude",
                "content_counts" := "content_counts"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "interests"
    FROM interests_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
interests_summary AS (
    SELECT "__row_position",
        count(*) AS "interests_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "interests_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "interests_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "interests_count_365d",
        min("age_days") AS "interests_last_age_days",
        count(DISTINCT "event_id") AS "interests_distinct_event_id",
        count(DISTINCT "organizer_id") AS "interests_distinct_organizer_id",
        count(TRY_CAST(CAST("invited" AS VARCHAR) AS BOOLEAN)) AS "interests_invited_known_count",
        CAST(count_if(TRY_CAST(CAST("invited" AS VARCHAR) AS BOOLEAN)) AS BIGINT) AS "interests_invited_count",
        CAST(count_if(TRY_CAST(CAST("invited" AS VARCHAR) AS BOOLEAN)) AS BIGINT) / nullif(count(TRY_CAST(CAST("invited" AS VARCHAR) AS BOOLEAN)), 0) AS "interests_invited_rate",
        count(TRY_CAST(CAST("interested" AS VARCHAR) AS BOOLEAN)) AS "interests_interested_known_count",
        CAST(count_if(TRY_CAST(CAST("interested" AS VARCHAR) AS BOOLEAN)) AS BIGINT) AS "interests_interested_count",
        CAST(count_if(TRY_CAST(CAST("interested" AS VARCHAR) AS BOOLEAN)) AS BIGINT) / nullif(count(TRY_CAST(CAST("interested" AS VARCHAR) AS BOOLEAN)), 0) AS "interests_interested_rate",
        count(TRY_CAST(CAST("not_interested" AS VARCHAR) AS BOOLEAN)) AS "interests_not_interested_known_count",
        CAST(count_if(TRY_CAST(CAST("not_interested" AS VARCHAR) AS BOOLEAN)) AS BIGINT) AS "interests_not_interested_count",
        CAST(count_if(TRY_CAST(CAST("not_interested" AS VARCHAR) AS BOOLEAN)) AS BIGINT) / nullif(count(TRY_CAST(CAST("not_interested" AS VARCHAR) AS BOOLEAN)), 0) AS "interests_not_interested_rate"
    FROM interests_ranked
    GROUP BY "__row_position"
),
organized_events_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        event."start_time" AS "__event_time",
        date_diff('microseconds', event."start_time", anchor."__as_of") / 86400000000.0 AS "age_days",
        CASE WHEN event."event_id" IS NOT NULL THEN 'rel-event:event:' || CAST(event."event_id" AS VARCHAR) END AS "event_id",
        event."start_time" AS "event_start",
        event."city" AS "city",
        event."state" AS "state",
        CASE WHEN event."zip" IS NOT NULL THEN 'rel-event:postal:' || CAST(event."zip" AS VARCHAR) END AS "zip_code",
        event."country" AS "country",
        TRY_CAST(event."lat" AS DOUBLE) AS "latitude",
        TRY_CAST(event."lng" AS DOUBLE) AS "longitude",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."start_time" DESC NULLS LAST, event."__source_row_id" DESC) AS "__recency",
        event."content_counts" AS "content_counts"
    FROM anchors AS anchor
    INNER JOIN event_content AS event ON anchor."__entity_key" = event."user_id" AND event."start_time" <= anchor."__as_of"
),
organized_events_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "age_days" := "age_days",
                "event_id" := "event_id",
                "event_start" := "event_start",
                "city" := "city",
                "state" := "state",
                "zip_code" := "zip_code",
                "country" := "country",
                "latitude" := "latitude",
                "longitude" := "longitude",
                "content_counts" := "content_counts"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC) AS "organized_events"
    FROM organized_events_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
-- Friendship edges have no timestamp; retained source order has no recency meaning.
friends_ranked AS (
    SELECT anchor."__row_position", event."__source_row_id" AS "__source_order",
        CASE WHEN event."friend" IS NOT NULL THEN 'rel-event:user:' || CAST(event."friend" AS VARCHAR) END AS "friend_id",
        details."locale" AS "locale",
        details."gender" AS "gender",
        details."location" AS "location",
        TRY_CAST(details."timezone" AS DOUBLE) AS "timezone",
        extract(year FROM anchor."__as_of") - TRY_CAST(details."birthyear" AS DOUBLE) AS "age_years",
        date_diff('microseconds', details."joinedAt", anchor."__as_of") / 86400000000.0 AS "tenure_days",
        row_number() OVER (PARTITION BY anchor."__row_position" ORDER BY event."__source_row_id" DESC) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "user_friends" AS event ON anchor."__entity_key" = event."user"
    LEFT JOIN LATERAL (SELECT linked.* FROM "users" AS linked WHERE linked."user_id" = event."friend" AND linked."joinedAt" <= anchor."__as_of" ORDER BY linked."joinedAt" DESC NULLS LAST, linked."__source_row_id" DESC LIMIT 1) AS details ON TRUE
),
organized_events_summary AS (
    SELECT "__row_position",
        count(*) AS "organized_events_count",
        CAST(count_if("age_days" <= 30) AS BIGINT) AS "organized_events_count_30d",
        CAST(count_if("age_days" <= 90) AS BIGINT) AS "organized_events_count_90d",
        CAST(count_if("age_days" <= 365) AS BIGINT) AS "organized_events_count_365d",
        min("age_days") AS "organized_events_last_age_days",
        count(DISTINCT "event_id") AS "organized_events_distinct_event_id"
    FROM organized_events_ranked
    GROUP BY "__row_position"
),
friends_grouped AS (
    SELECT "__row_position",
        list(struct_pack(
                "friend_id" := "friend_id",
                "locale" := "locale",
                "gender" := "gender",
                "location" := "location",
                "timezone" := "timezone",
                "age_years" := "age_years",
                "tenure_days" := "tenure_days"
            ) ORDER BY "__source_order" ASC) AS "friends"
    FROM friends_ranked
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
friends_summary AS (
    SELECT "__row_position",
        count(*) AS "friends_count",
        count(DISTINCT "friend_id") AS "friends_distinct_friend_id",
        count("tenure_days") AS "friends_profile_known_count"
    FROM friends_ranked
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce(attendance_grouped."attendance", []) AS "attendance",
    coalesce(attendance_summary."attendance_count", 0) AS "attendance_count",
    coalesce(attendance_summary."attendance_count_30d", 0) AS "attendance_count_30d",
    coalesce(attendance_summary."attendance_count_90d", 0) AS "attendance_count_90d",
    coalesce(attendance_summary."attendance_count_365d", 0) AS "attendance_count_365d",
    attendance_summary."attendance_last_age_days" AS "attendance_last_age_days",
    coalesce(attendance_summary."attendance_distinct_event_id", 0) AS "attendance_distinct_event_id",
    coalesce(attendance_summary."attendance_distinct_organizer_id", 0) AS "attendance_distinct_organizer_id",
    coalesce(attendance_summary."attendance_status_known_count", 0) AS "attendance_status_known_count",
    coalesce(attendance_summary."attendance_accepted_count", 0) AS "attendance_accepted_count",
    coalesce(attendance_summary."attendance_invited_count", 0) AS "attendance_invited_count",
    attendance_summary."attendance_accepted_rate" AS "attendance_accepted_rate",
    coalesce(attendance_summary."attendance_repeated_event_count", 0) AS "attendance_repeated_event_count",
    coalesce(interests_grouped."interests", []) AS "interests",
    coalesce(interests_summary."interests_count", 0) AS "interests_count",
    coalesce(interests_summary."interests_count_30d", 0) AS "interests_count_30d",
    coalesce(interests_summary."interests_count_90d", 0) AS "interests_count_90d",
    coalesce(interests_summary."interests_count_365d", 0) AS "interests_count_365d",
    interests_summary."interests_last_age_days" AS "interests_last_age_days",
    coalesce(interests_summary."interests_distinct_event_id", 0) AS "interests_distinct_event_id",
    coalesce(interests_summary."interests_distinct_organizer_id", 0) AS "interests_distinct_organizer_id",
    coalesce(interests_summary."interests_invited_known_count", 0) AS "interests_invited_known_count",
    coalesce(interests_summary."interests_invited_count", 0) AS "interests_invited_count",
    interests_summary."interests_invited_rate" AS "interests_invited_rate",
    coalesce(interests_summary."interests_interested_known_count", 0) AS "interests_interested_known_count",
    coalesce(interests_summary."interests_interested_count", 0) AS "interests_interested_count",
    interests_summary."interests_interested_rate" AS "interests_interested_rate",
    coalesce(interests_summary."interests_not_interested_known_count", 0) AS "interests_not_interested_known_count",
    coalesce(interests_summary."interests_not_interested_count", 0) AS "interests_not_interested_count",
    interests_summary."interests_not_interested_rate" AS "interests_not_interested_rate",
    coalesce(organized_events_grouped."organized_events", []) AS "organized_events",
    coalesce(organized_events_summary."organized_events_count", 0) AS "organized_events_count",
    coalesce(organized_events_summary."organized_events_count_30d", 0) AS "organized_events_count_30d",
    coalesce(organized_events_summary."organized_events_count_90d", 0) AS "organized_events_count_90d",
    coalesce(organized_events_summary."organized_events_count_365d", 0) AS "organized_events_count_365d",
    organized_events_summary."organized_events_last_age_days" AS "organized_events_last_age_days",
    coalesce(organized_events_summary."organized_events_distinct_event_id", 0) AS "organized_events_distinct_event_id",
    coalesce(friends_grouped."friends", []) AS "friends",
    coalesce(friends_summary."friends_count", 0) AS "friends_count",
    coalesce(friends_summary."friends_distinct_friend_id", 0) AS "friends_distinct_friend_id",
    coalesce(friends_summary."friends_profile_known_count", 0) AS "friends_profile_known_count"
FROM root_features
LEFT JOIN attendance_grouped USING ("__row_position")
LEFT JOIN attendance_summary USING ("__row_position")
LEFT JOIN interests_grouped USING ("__row_position")
LEFT JOIN interests_summary USING ("__row_position")
LEFT JOIN organized_events_grouped USING ("__row_position")
LEFT JOIN organized_events_summary USING ("__row_position")
LEFT JOIN friends_grouped USING ("__row_position")
LEFT JOIN friends_summary USING ("__row_position")
ORDER BY "__row_position";
