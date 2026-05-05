import pandas as pd
import numpy as np
import yaml
import os


def get_candidates(tweets_df, k=100):
    # Recent + random mix
    recent = tweets_df.sort_values("created_at", ascending=False).head(50)
    random_sample = tweets_df.sample(50)

    return pd.concat([recent, random_sample]).drop_duplicates().head(k)


def run():
    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Load raw data
    users_df = pd.read_csv("data/raw/users.csv")
    tweets_df = pd.read_csv("data/raw/tweets.csv")
    authors_df = pd.read_csv("data/raw/authors.csv")
    history_df = pd.read_csv("data/raw/user_author.csv")

    tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"])

    all_rows = []

    # 🔥 LOOP ONLY OVER USERS (NOT tweets)
    for _, user in users_df.iterrows():

        candidates = get_candidates(tweets_df)

        # Create user-tweet pairs
        user_df = candidates.copy()
        user_df["user_id"] = user["user_id"]
        user_df["preferred_topics"] = str(user["preferred_topics"])
        user_df["user_location"] = user["location"]

        all_rows.append(user_df)

    base_df = pd.concat(all_rows, ignore_index=True)

    # 🔥 JOIN AUTHOR FEATURES
    base_df = base_df.merge(authors_df, on="author_id", how="left")

    # 🔥 JOIN USER-AUTHOR HISTORY
    base_df = base_df.merge(history_df, on=["user_id", "author_id"], how="left")

    # Fill missing history
    base_df["engagement_3m"] = base_df["engagement_3m"].fillna(0)
    base_df["engagement_1y"] = base_df["engagement_1y"].fillna(0)

    # Save
    os.makedirs("data/interim", exist_ok=True)
    base_df.to_csv("data/interim/base_df.csv", index=False)

    print("✅ prepare_data complete")
    print(base_df.shape)


if __name__ == "__main__":
    run()