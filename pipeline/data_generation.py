import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta


def generate_users(params):
    users = []
    for i in range(params["data"]["num_users"]):
        users.append({
            "user_id": f"U{i}",
            "preferred_topics": random.sample(params["topics"], 2),
            "location": random.choice(params["locations"])
        })
    return pd.DataFrame(users)


def generate_authors(params):
    authors = []
    for i in range(params["data"]["num_authors"]):
        followers = np.random.randint(100, 1_000_000)

        authors.append({
            "author_id": f"A{i}",
            "is_verified": np.random.choice([0, 1], p=[0.8, 0.2]),
            "num_followers": followers,
            "social_rank": np.log1p(followers)
        })

    return pd.DataFrame(authors)


def generate_tweets(params, authors_df):
    tweets = []
    base_time = datetime.now()

    for i in range(params["data"]["num_tweets"]):
        topic = random.choice(params["topics"])
        created_at = base_time - timedelta(hours=np.random.randint(1, 72))

        tweets.append({
            "tweet_id": f"T{i}",
            "author_id": random.choice(authors_df["author_id"].values),
            "topic": topic,
            "tweet_length": np.random.randint(5, 30),
            "has_media": np.random.choice([0, 1]),
            "is_url": np.random.choice([0, 1]),
            "created_at": created_at
        })

    return pd.DataFrame(tweets)


def generate_user_author_history_df(users_df, authors_df):
    rows = []

    for _, user in users_df.iterrows():
        for _, author in authors_df.iterrows():
            rows.append({
                "user_id": user["user_id"],
                "author_id": author["author_id"],
                "engagement_3m": np.random.poisson(2),
                "engagement_1y": np.random.poisson(5)
            })

    return pd.DataFrame(rows)