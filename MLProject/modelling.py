import os
import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from mlflow.models.signature import infer_signature


TARGET_COLUMN = "Churn"
DATA_DIR = "telco_customer_churn_preprocessing"
TRAIN_PATH = os.path.join(DATA_DIR, "train_preprocessed.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_preprocessed.csv")
ARTIFACT_DIR = "artifacts"


def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow Tracking URI: {tracking_uri}")


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def create_artifact_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def save_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred)

    path = os.path.join(ARTIFACT_DIR, "classification_report.txt")

    with open(path, "w") as file:
        file.write(report)

    return path


def save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    display.plot()
    plt.title("Confusion Matrix - Telco Customer Churn CI")
    plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"))
    plt.close()


def save_metrics(metrics):
    path = os.path.join(ARTIFACT_DIR, "metrics.json")

    with open(path, "w") as file:
        json.dump(metrics, file, indent=4)

    return path


def save_best_params(best_params):
    path = os.path.join(ARTIFACT_DIR, "best_params.json")

    with open(path, "w") as file:
        json.dump(best_params, file, indent=4)

    return path


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def save_run_id(run_id):
    workspace_dir = os.getenv("GITHUB_WORKSPACE", os.getcwd())
    run_id_path = os.path.join(workspace_dir, "latest_run_id.txt")

    with open(run_id_path, "w") as file:
        file.write(run_id)

    print(f"Run ID saved to: {run_id_path}")


def main():
    setup_mlflow()
    create_artifact_dir()

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.runName", "RandomForest_CI_Retraining")

        grid_search = train_model(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))
        best_cv_score = float(grid_search.best_score_)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "best_cv_score": best_cv_score
        }

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("cv", 3)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metrics(metrics)

        save_classification_report(y_test, y_pred)
        save_confusion_matrix(y_test, y_pred)
        save_metrics(metrics)
        save_best_params(grid_search.best_params_)

        mlflow.log_artifacts(ARTIFACT_DIR)

        input_example = X_test.head(5)
        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
            pip_requirements=[
                "mlflow==2.18.0",
                "pandas",
                "numpy",
                "scikit-learn",
                "cloudpickle"
            ]
        )

        save_run_id(run.info.run_id)

        print("Training CI selesai.")
        print("Run ID:", run.info.run_id)
        print("Best Params:", grid_search.best_params_)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC:", roc_auc)


if __name__ == "__main__":
    main()