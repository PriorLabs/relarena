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
        root."PostTypeId" AS "post_type",
        root."OwnerUserId" AS "owner_user_id",
        root."ParentId" AS "parent_id",
        root."Title" AS "title",
        root."Body" AS "body",
        root."Tags" AS "tags"
    FROM anchors AS anchor
    LEFT JOIN "posts" AS root
        ON anchor."__entity_key" = root."Id"
        AND root."CreationDate" <= anchor."__as_of"
    QUALIFY row_number() OVER (
        PARTITION BY anchor."__row_position"
        ORDER BY root."CreationDate" DESC NULLS LAST, root."__source_row_id" DESC NULLS LAST
    ) = 1
),
"votes_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "__event_time",
        event."UserId" AS "user_id",
        event."VoteTypeId" AS "vote_type",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."CreationDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "votes" AS event
        ON anchor."__entity_key" = event."PostId"
        AND event."CreationDate" <= anchor."__as_of"
),
"votes_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "user_id" := "user_id",
                "vote_type" := "vote_type"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "votes"
    FROM "votes_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"comments_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "__event_time",
        event."UserId" AS "user_id",
        event."Text" AS "comment_text",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."CreationDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "comments" AS event
        ON anchor."__entity_key" = event."PostId"
        AND event."CreationDate" <= anchor."__as_of"
),
"comments_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "user_id" := "user_id",
                "comment_text" := "comment_text"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "comments"
    FROM "comments_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"revisions_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "__event_time",
        event."UserId" AS "user_id",
        event."PostHistoryTypeId" AS "revision_type",
        event."Comment" AS "revision_comment",
        event."Text" AS "revision_text",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."CreationDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "postHistory" AS event
        ON anchor."__entity_key" = event."PostId"
        AND event."CreationDate" <= anchor."__as_of"
),
"revisions_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "user_id" := "user_id",
                "revision_type" := "revision_type",
                "revision_comment" := "revision_comment",
                "revision_text" := "revision_text"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "revisions"
    FROM "revisions_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
),
"links_ranked" AS (
    SELECT
        anchor."__row_position",
        anchor."__as_of" AS "__anchor_time",
        event."__source_row_id" AS "__source_order",
        event."CreationDate" AS "__event_time",
        event."RelatedPostId" AS "related_post_id",
        event."LinkTypeId" AS "link_type",
        row_number() OVER (
            PARTITION BY anchor."__row_position"
            ORDER BY event."CreationDate" DESC NULLS LAST, event."__source_row_id" DESC
        ) AS "__recency"
    FROM anchors AS anchor
    INNER JOIN "postLinks" AS event
        ON anchor."__entity_key" = event."PostId"
        AND event."CreationDate" <= anchor."__as_of"
),
"links_grouped" AS (
    SELECT
        "__row_position",
        list(
            struct_pack(
                "age_days" := date_diff('microseconds', "__event_time", "__anchor_time") / 86400000000.0,
                "related_post_id" := "related_post_id",
                "link_type" := "link_type"
            ) ORDER BY "__event_time" ASC NULLS LAST, "__source_order" ASC
        ) AS "links"
    FROM "links_ranked"
    WHERE "__recency" <= 64
    GROUP BY "__row_position"
)
SELECT
    root_features.*,
    coalesce("votes_grouped"."votes", []) AS "votes",
    coalesce("comments_grouped"."comments", []) AS "comments",
    coalesce("revisions_grouped"."revisions", []) AS "revisions",
    coalesce("links_grouped"."links", []) AS "links"
FROM root_features
LEFT JOIN "votes_grouped" USING ("__row_position")
LEFT JOIN "comments_grouped" USING ("__row_position")
LEFT JOIN "revisions_grouped" USING ("__row_position")
LEFT JOIN "links_grouped" USING ("__row_position")
ORDER BY "__row_position";
