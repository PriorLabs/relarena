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
        root."user_id" AS "entity_id",
        root."locale" AS "locale",
        root."birthyear" AS "birthyear",
        root."gender" AS "gender",
        root."location" AS "location",
        root."timezone" AS "timezone",
        root."joinedAt" AS "joined_at"
    FROM anchors AS anchor
    LEFT JOIN "users" AS root
        ON anchor."__entity_key" = root."user_id"
        AND root."joinedAt" <= anchor."__as_of"
    QUALIFY row_number() OVER (
        PARTITION BY anchor."__row_position"
        ORDER BY root."joinedAt" DESC NULLS LAST, root."__source_row_id" DESC NULLS LAST
    ) = 1
),
"attendance_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."start_time" AS "__event_time",
        event."event" AS "event_id",
        event."start_time" AS "event_start",
        event."status" AS "status",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."start_time" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "event_attendees" AS event
        ON anchor."__entity_key" = event."user_id"
        AND event."start_time" <= anchor."__as_of"
),
"attendance_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "event_id" := "event_id",
                "event_start" := "event_start",
                "status" := "status"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "attendance"
    FROM "attendance_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"interests_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."timestamp" AS "__event_time",
        event."event" AS "event_id",
        event."timestamp" AS "response_time",
        event."invited" AS "invited",
        event."interested" AS "interested",
        event."not_interested" AS "not_interested",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."timestamp" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "event_interest" AS event
        ON anchor."__entity_key" = event."user"
        AND event."timestamp" <= anchor."__as_of"
),
"interests_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "event_id" := "event_id",
                "response_time" := "response_time",
                "invited" := "invited",
                "interested" := "interested",
                "not_interested" := "not_interested"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "interests"
    FROM "interests_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"organized_events_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."start_time" AS "__event_time",
        event."event_id" AS "event_id",
        event."start_time" AS "event_start",
        event."city" AS "city",
        event."state" AS "state",
        event."zip" AS "zip_code",
        event."country" AS "country",
        event."lat" AS "latitude",
        event."lng" AS "longitude",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."start_time" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "events" AS event
        ON anchor."__entity_key" = event."user_id"
        AND event."start_time" <= anchor."__as_of"
),
"organized_events_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "event_id" := "event_id",
                "event_start" := "event_start",
                "city" := "city",
                "state" := "state",
                "zip_code" := "zip_code",
                "country" := "country",
                "latitude" := "latitude",
                "longitude" := "longitude"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "organized_events"
    FROM "organized_events_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"friends_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."friend" AS "friend_id",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "user_friends" AS event
        ON anchor."__entity_key" = event."user"
),
"friends_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "friend_id" := "friend_id"
            ) ORDER BY "__source_order" ASC
        ) AS "friends"
    FROM "friends_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("attendance_grouped"."attendance", []) AS "attendance",
    coalesce("interests_grouped"."interests", []) AS "interests",
    coalesce("organized_events_grouped"."organized_events", []) AS "organized_events",
    coalesce("friends_grouped"."friends", []) AS "friends"
FROM root_features
LEFT JOIN "attendance_grouped" USING ("__row_position")
LEFT JOIN "interests_grouped" USING ("__row_position")
LEFT JOIN "organized_events_grouped" USING ("__row_position")
LEFT JOIN "friends_grouped" USING ("__row_position")
ORDER BY "__row_position";
