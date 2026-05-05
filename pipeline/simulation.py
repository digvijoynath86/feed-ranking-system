import numpy as np
import pandas as pd


# -------------------------
# TRUE USER BEHAVIOR MODEL
# -------------------------
def true_engagement_prob(df: pd.DataFrame) -> pd.Series:

    prob = 0.05

    prob += 0.25 * df["topic_similarity"]
    prob += 0.05 * df["engagement_3m"]
    prob += 0.03 * df["engagement_1y"]
    prob += 0.05 * df["is_verified"]
    prob += 0.02 * df["social_rank"]
    prob += 0.05 * df["has_media"]
    prob += 0.03 * (1 - df["is_url"])
    prob += 0.2 * np.exp(-df["recency_hours"] / 24)

    # Noise
    prob += np.random.normal(0, 0.1, size=len(df))

    return np.clip(prob, 0, 1)


# -------------------------
# INITIAL RANKING (RANDOM)
# -------------------------
def initial_rank(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["init_score"] = np.random.rand(len(df))

    return df.sort_values(["user_id", "init_score"], ascending=[True, False])


# -------------------------
# SIMULATE IMPRESSIONS
# -------------------------
def simulate_impressions(df: pd.DataFrame, top_k=10) -> pd.DataFrame:

    df = initial_rank(df)

    df["rank"] = df.groupby("user_id").cumcount() + 1

    impressions = df[df["rank"] <= top_k].copy()

    # Position bias
    impressions["position_bias"] = 1 / np.log2(impressions["rank"] + 1)

    return impressions


# -------------------------
# SIMULATE USER ACTIONS
# -------------------------
def simulate_actions(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    true_prob = true_engagement_prob(df)

    final_prob = true_prob * df["position_bias"]

    df["engagement"] = np.random.binomial(1, final_prob)

    return df


# -------------------------
# FULL PIPELINE
# -------------------------
def simulate_logs(df: pd.DataFrame) -> pd.DataFrame:

    impressions = simulate_impressions(df)
    logs = simulate_actions(impressions)

    return logs