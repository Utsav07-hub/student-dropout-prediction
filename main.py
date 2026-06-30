import mlflow
import mlflow.sklearn
from sklearn.metrics import ConfusionMatrixDisplay
mlflow.set_tracking_uri("sqlite:///mlflow.db") 
mlflow.set_experiment("Student Dropout Prediction")

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt 

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_ingestion import load_data
from src.evaluate import evaluate_model
from src.preprocessing import preprocess_data
from src.train import train_models, tune_random_forest
from src.utils import save_csv, save_model
from src.logger import logger


DATA_PATH = Path("data/raw/student_dropout_dataset.csv")
MODEL_PATH = Path("models/best_model.pkl")
METRICS_PATH = Path("artifacts/metrics/metrics.csv")


def main() -> None:
    logger.info("Loading dataset...")
    df = load_data(DATA_PATH)

    logger.info("Preprocessing data...")
    processed = preprocess_data(df)
    
    save_model(processed["preprocessor"], "models/preprocessor.pkl")
    save_model(processed["label_encoder"], "models/label_encoder.pkl")

    X_train = processed["X_train"]
    X_test = processed["X_test"]
    y_train = processed["y_train"]
    y_test = processed["y_test"]

    logger.info("Training baseline models...")
    trained_models = train_models(X_train, y_train)

    logger.info("Tuning best model...")
    best_model = tune_random_forest(X_train, y_train)

    logger.info("Evaluating the tuned model...")
    metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)

    logger.info("Saving model and metrics...")
    save_model(best_model, MODEL_PATH)
    save_csv(pd.DataFrame([metrics]), METRICS_PATH)

    with mlflow.start_run(run_name="Random Forest Tuned_version_1.0"):
        # mlflow.log_param("model", "Random Forest")
        # mlflow.log_param("n_estimators", best_model.n_estimators)
        # mlflow.log_param("max_depth", best_model.max_depth)
        # mlflow.log_param("min_samples_split", best_model.min_samples_split)
        # mlflow.log_param("min_samples_leaf", best_model.min_samples_leaf)
        mlflow.log_params(best_model.get_params())
        mlflow.log_params(best_model.get_params())
        
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "cv_score": float(metrics["cv_score"])
        })
        
        ConfusionMatrixDisplay.from_estimator(
            best_model,
            X_test,
            y_test,
            cmap="Blues"
        )

        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")

        mlflow.log_artifact("confusion_matrix.png")

        plt.close()

        mlflow.sklearn.log_model(sk_model=best_model, name ="random_forest_model")

    logger.info("Training completed successfully.")
    print("\nEvaluation Metrics:")
    print(metrics)
    
if __name__ == "__main__":
    main()

# mlflow ui --backend-store-uri sqlite:///mlflow.db