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
        root."Id" AS "entity_id",
        root."CreationDate" AS "created_at",
        root."DisplayName" AS "display_name",
        root."WebsiteUrl" AS "website_url",
        root."Location" AS "location",
        root."AboutMe" AS "about_me",
        CAST(root."AccountId" AS BIGINT) AS "account_id"
    FROM anchors AS anchor
    LEFT JOIN "users" AS root
        ON anchor."__entity_key" = root."Id"
        AND root."CreationDate" <= anchor."__as_of"
    QUALIFY row_number() OVER (
        PARTITION BY anchor."__row_position"
        ORDER BY root."CreationDate" DESC NULLS LAST, root."__source_row_id" DESC NULLS LAST
    ) = 1
),
"badges_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."Date" AS "__event_time",
        event."Name" AS "badge_name",
        event."Class" AS "badge_class",
        event."TagBased" AS "tag_based",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."Date" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "badges" AS event
        ON anchor."__entity_key" = event."UserId"
        AND event."Date" <= anchor."__as_of"
),
"badges_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "badge_name" := "badge_name",
                "badge_class" := "badge_class",
                "tag_based" := "tag_based"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "badges"
    FROM "badges_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"posts_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "__event_time",
        event."Id" AS "post_id",
        event."PostTypeId" AS "post_type",
        event."ParentId" AS "parent_id",
        event."Title" AS "title",
        event."Body" AS "body",
        event."Tags" AS "tags",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."CreationDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "posts" AS event
        ON anchor."__entity_key" = event."OwnerUserId"
        AND event."CreationDate" <= anchor."__as_of"
),
"posts_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "post_id" := "post_id",
                "post_type" := "post_type",
                "parent_id" := "parent_id",
                "title" := "title",
                "body" := "body",
                "tags" := "tags"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "posts"
    FROM "posts_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"comments_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "__event_time",
        event."PostId" AS "post_id",
        event."Text" AS "comment_text",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."CreationDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "comments" AS event
        ON anchor."__entity_key" = event."UserId"
        AND event."CreationDate" <= anchor."__as_of"
),
"comments_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "post_id" := "post_id",
                "comment_text" := "comment_text"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "comments"
    FROM "comments_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"votes_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "__event_time",
        event."PostId" AS "post_id",
        event."VoteTypeId" AS "vote_type",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."CreationDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "votes" AS event
        ON anchor."__entity_key" = event."UserId"
        AND event."CreationDate" <= anchor."__as_of"
),
"votes_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "post_id" := "post_id",
                "vote_type" := "vote_type"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "votes"
    FROM "votes_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("badges_grouped"."badges", []) AS "badges",
    coalesce("posts_grouped"."posts", []) AS "posts",
    coalesce("comments_grouped"."comments", []) AS "comments",
    coalesce("votes_grouped"."votes", []) AS "votes"
FROM root_features
LEFT JOIN "badges_grouped" USING ("__row_position")
LEFT JOIN "posts_grouped" USING ("__row_position")
LEFT JOIN "comments_grouped" USING ("__row_position")
LEFT JOIN "votes_grouped" USING ("__row_position")
ORDER BY "__row_position";
