import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:

    df = df.copy()

    # -------------------------
    # USER-TWEET FEATURE
    # -------------------------
    # preferred_topics is string → convert to list
    df["preferred_topics"] = df["preferred_topics"].apply(eval)

    df["topic_similarity"] = df.apply(
        lambda row: 1 if row["topic"] in row["preferred_topics"] else 0,
        axis=1
    )

    # -------------------------
    # RECENCY FEATURE
    # -------------------------
    current_time = pd.Timestamp.now()

    df["created_at"] = pd.to_datetime(df["created_at"])
    df["recency_hours"] = (current_time - df["created_at"]).dt.total_seconds() / 3600

    # -------------------------
    # TIME FEATURES
    # -------------------------
    df["day_of_week"] = df["created_at"].dt.weekday

    hour = df["created_at"].dt.hour

    df["time_of_day"] = np.select(
        [
            (hour >= 6) & (hour < 12),
            (hour >= 12) & (hour < 18),
            (hour >= 18) & (hour < 24)
        ],
        [0, 1, 2],  # morning, afternoon, evening
        default=3   # night
    )

    # -------------------------
    # RANDOM CONTEXT FEATURE
    # -------------------------
    df["holiday_approaching"] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

    # -------------------------
    # CLEANUP / SELECT FEATURES
    # -------------------------
    feature_cols = [
        "user_id",
        "tweet_id",
        "author_id",

        # USER-AUTHOR
        "engagement_3m",
        "engagement_1y",

        # AUTHOR
        "is_verified",
        "num_followers",
        "social_rank",

        # USER-TWEET
        "topic_similarity",

        # TWEET
        "tweet_length",
        "recency_hours",
        "has_media",
        "is_url",

        # CONTEXT
        "day_of_week",
        "time_of_day",
        "holiday_approaching",
        "user_location"
    ]

    return df[feature_cols]