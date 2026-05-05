import pandas as pd
import yaml
import joblib
import os
import mlflow
import mlflow.lightgbm

from pipeline.ranker import prepare_data, train_model, evaluate_ndcg


def run():

    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Load logs
    df = pd.read_csv("data/processed/logs_df.csv")

    # Prepare ranking groups
    train_df, test_df, train_group, test_group = prepare_data(df)

    mlflow.set_experiment("feed-ranking")

    with mlflow.start_run():

        model = train_model(train_df, test_df, train_group, test_group, params)

        train_ndcg = evaluate_ndcg(train_df, model)
        test_ndcg = evaluate_ndcg(test_df, model)

        # Log metrics
        mlflow.log_metric("train_ndcg", train_ndcg)
        mlflow.log_metric("test_ndcg", test_ndcg)

        # Log params
        mlflow.log_params(params["model"])

        # Log model
        mlflow.lightgbm.log_model(model, "model")

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/ranker.pkl")

        print("✅ Training complete")
        print("Train NDCG:", train_ndcg)
        print("Test NDCG:", test_ndcg)


if __name__ == "__main__":
    run()