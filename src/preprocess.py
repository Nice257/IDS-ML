"""
Prétraitement du dataset NSL-KDD pour un système IDS basé sur le Machine Learning.

Ce script charge les fichiers KDDTrain+.txt et KDDTest+.txt, encode les variables
catégorielles, transforme la cible en classification binaire, normalise les
variables numériques et sauvegarde les jeux prêts pour l'entraînement.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

FEATURES_OUTPUT = DATA_DIR / "processed_nsl_kdd.joblib"
PREPROCESSOR_OUTPUT = MODELS_DIR / "preprocessor.joblib"

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
TARGET_COLUMN = "binary_label"

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Charge un fichier NSL-KDD et applique les noms de colonnes officiels."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}. "
            "Placez KDDTrain+.txt et KDDTest+.txt dans le dossier data/."
        )

    return pd.read_csv(file_path, names=NSL_KDD_COLUMNS)


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Crée une cible binaire : 0 pour normal, 1 pour attaque."""
    prepared_df = df.copy()
    prepared_df[TARGET_COLUMN] = np.where(prepared_df["label"] == "normal", 0, 1)
    return prepared_df.drop(columns=["label", "difficulty"])


def fit_label_encoders(train_df: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Entraîne un LabelEncoder pour chaque colonne catégorielle du jeu train."""
    encoders = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoder.fit(train_df[column].astype(str))
        encoders[column] = encoder

    return encoders


def transform_with_unknowns(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    """Encode les colonnes catégorielles en gérant les valeurs inconnues."""
    transformed_df = df.copy()

    for column, encoder in encoders.items():
        known_classes = set(encoder.classes_)
        values = transformed_df[column].astype(str)
        fallback_class = encoder.classes_[0]
        safe_values = values.where(values.isin(known_classes), fallback_class)
        transformed_df[column] = encoder.transform(safe_values)

    return transformed_df


def prepare_input_features(df: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    """Prépare un CSV de trafic réseau pour une prédiction avec un modèle sauvegardé."""
    input_df = df.copy()
    feature_names = preprocessor["feature_names"]

    for column in ["label", "difficulty", TARGET_COLUMN]:
        if column in input_df.columns:
            input_df = input_df.drop(columns=[column])

    missing_columns = [column for column in feature_names if column not in input_df.columns]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le fichier uploadé : "
            + ", ".join(missing_columns)
        )

    input_df = input_df[feature_names]
    encoded_df = transform_with_unknowns(input_df, preprocessor["encoders"])
    return preprocessor["scaler"].transform(encoded_df)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les variables explicatives X et la cible y."""
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return x, y


def scale_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Normalise les variables avec StandardScaler entraîné uniquement sur train."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def save_processed_data(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> None:
    """Sauvegarde les jeux prétraités dans data/ au format joblib."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "X_train": x_train,
            "X_test": x_test,
            "y_train": y_train.to_numpy(),
            "y_test": y_test.to_numpy(),
            "feature_names": feature_names,
        },
        FEATURES_OUTPUT,
    )


def save_preprocessor(
    encoders: dict[str, LabelEncoder], scaler: StandardScaler, feature_names: list[str]
) -> None:
    """Sauvegarde les encodeurs et le scaler pour réutilisation en production."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "encoders": encoders,
            "scaler": scaler,
            "feature_names": feature_names,
            "categorical_columns": CATEGORICAL_COLUMNS,
        },
        PREPROCESSOR_OUTPUT,
    )


def preprocess_data() -> None:
    """Exécute tout le pipeline de prétraitement NSL-KDD."""
    train_df = create_binary_target(load_dataset(TRAIN_FILE))
    test_df = create_binary_target(load_dataset(TEST_FILE))

    encoders = fit_label_encoders(train_df)
    train_encoded = transform_with_unknowns(train_df, encoders)
    test_encoded = transform_with_unknowns(test_df, encoders)

    x_train, y_train = split_features_target(train_encoded)
    x_test, y_test = split_features_target(test_encoded)

    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)
    feature_names = x_train.columns.tolist()

    save_processed_data(x_train_scaled, x_test_scaled, y_train, y_test, feature_names)
    save_preprocessor(encoders, scaler, feature_names)

    print(f"Données prétraitées sauvegardées dans : {FEATURES_OUTPUT}")
    print(f"Préprocesseur sauvegardé dans : {PREPROCESSOR_OUTPUT}")
    print(f"Train : {x_train_scaled.shape}, Test : {x_test_scaled.shape}")


def main() -> None:
    """Point d'entrée principal du script."""
    preprocess_data()


if __name__ == "__main__":
    main()
