import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score


FEATURES = [
    "engagement_3m",
    "engagement_1y",
    "is_verified",
    "social_rank",
    "topic_similarity",
    "tweet_length",
    "recency_hours",
    "has_media",
    "is_url",
    "day_of_week",
    "time_of_day",
    "holiday_approaching"
]

TARGET = "engagement"


def prepare_data(df):
    users = df["user_id"].unique()

    train_users, test_users = train_test_split(users, test_size=0.2, random_state=42)

    train_df = df[df["user_id"].isin(train_users)]
    test_df = df[df["user_id"].isin(test_users)]

    train_group = train_df.groupby("user_id").size().values
    test_group = test_df.groupby("user_id").size().values

    return train_df, test_df, train_group, test_group


def train_model(train_df, test_df, train_group, test_group, params):

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=params["model"]["n_estimators"],
        learning_rate=params["model"]["learning_rate"],
        num_leaves=params["model"]["num_leaves"]
    )

    model.fit(
        train_df[FEATURES],
        train_df[TARGET],
        group=train_group,
        eval_set=[(test_df[FEATURES], test_df[TARGET])],
        eval_group=[test_group],
        eval_at=[10]
    )

    return model


def evaluate_ndcg(df, model):

    scores = []

    for user_id, group_df in df.groupby("user_id"):
        y_true = group_df[TARGET].values.reshape(1, -1)
        y_pred = model.predict(group_df[FEATURES]).reshape(1, -1)

        scores.append(ndcg_score(y_true, y_pred, k=10))

    return float(np.mean(scores))