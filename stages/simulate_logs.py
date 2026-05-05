import pandas as pd
import os

from pipeline.simulation import simulate_logs


def run():

    # Load features
    df = pd.read_csv("data/processed/features.csv")

    # Generate logs
    logs_df = simulate_logs(df)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    logs_df.to_csv("data/processed/logs_df.csv", index=False)

    print("✅ Logs simulation complete")
    print(logs_df.shape)


if __name__ == "__main__":
    run()