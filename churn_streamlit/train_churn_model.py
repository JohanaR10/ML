from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_preprocessing import preprocess_raw_data, split_features_target


def read_input_csv(path: Path, sep: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, encoding=encoding)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train(csv_path: Path, output_path: Path, sep: str, encoding: str) -> None:
    raw_df = read_input_csv(csv_path, sep=sep, encoding=encoding)
    processed_df = preprocess_raw_data(raw_df, training=True)
    X, y = split_features_target(processed_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, probabilities)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "threshold": 0.5,
        "metrics": {
            "roc_auc": float(roc_auc),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"Modelo guardado en: {output_path}")
    print(f"ROC AUC prueba: {roc_auc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de churn para Streamlit.")
    parser.add_argument("--csv", required=True, type=Path, help="Ruta al CSV historico.")
    parser.add_argument("--output", default=Path("models/churn_model.joblib"), type=Path)
    parser.add_argument("--sep", default="|", help="Separador del CSV. En el notebook era '|'.")
    parser.add_argument("--encoding", default="utf-16", help="Encoding del CSV. En el notebook era utf-16.")
    args = parser.parse_args()

    train(args.csv, args.output, args.sep, args.encoding)


if __name__ == "__main__":
    main()
