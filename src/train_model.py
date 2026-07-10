"""Train and save a reproducible breast tumor classification pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "breast-cancer.csv.xls"
MODEL_PATH = BASE_DIR / "Breast_cancer_model.pkl"
TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the committed WDBC-format dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if dataframe.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    return dataframe


def prepare_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the feature matrix and encoded binary target."""
    target = dataframe[TARGET_COLUMN].map({"B": 0, "M": 1})

    if target.isna().any():
        raise ValueError("Diagnosis must contain only 'B' and 'M' values.")

    features = dataframe.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    return features, target.astype(int)


def build_pipeline() -> Pipeline:
    """Create a scaling and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    dataframe = load_dataset()
    features, target = prepare_features(dataframe)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    train_predictions = pipeline.predict(x_train)
    test_predictions = pipeline.predict(x_test)

    print(f"Training accuracy: {accuracy_score(y_train, train_predictions):.4f}")
    print(f"Testing accuracy:  {accuracy_score(y_test, test_predictions):.4f}")
    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Benign", "Malignant"],
            digits=4,
        )
    )

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(pipeline, model_file)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
