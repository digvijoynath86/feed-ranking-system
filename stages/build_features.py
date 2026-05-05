import pandas as pd
import yaml
import os

from pipeline.build_features import build_features


def run():

    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Load input
    df = pd.read_csv("data/interim/base_df.csv")

    # Build features
    df_features = build_features(df, params)

    # Save output
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_csv("data/processed/features.csv", index=False)

    print("✅ Feature engineering complete")
    print(df_features.shape)


if __name__ == "__main__":
    run()