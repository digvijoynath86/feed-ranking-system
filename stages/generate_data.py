import os
import yaml
import pandas as pd
import numpy as np
import random

from pipeline.data_generation import (
    generate_users,
    generate_authors,
    generate_tweets,
    generate_user_author_history_df
)


def run():
    # Reproducibility
    np.random.seed(42)
    random.seed(42)

    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Generate data
    users_df = generate_users(params)
    authors_df = generate_authors(params)
    tweets_df = generate_tweets(params, authors_df)

    user_author_history_df = generate_user_author_history_df(users_df, authors_df)

    # Ensure folder exists
    os.makedirs("data/raw", exist_ok=True)

    # Save datasets
    users_df.to_csv("data/raw/users.csv", index=False)
    authors_df.to_csv("data/raw/authors.csv", index=False)
    tweets_df.to_csv("data/raw/tweets.csv", index=False)
    user_author_history_df.to_csv("data/raw/user_author.csv", index=False)


    print("✅ Data generation complete")
    print("Users:", users_df.shape)
    print("Authors:", authors_df.shape)
    print("Tweets:", tweets_df.shape)


if __name__ == "__main__":
    run()