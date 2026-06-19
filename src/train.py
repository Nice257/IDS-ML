"""
Entraînement des modèles Machine Learning pour la détection d'intrusions NSL-KDD.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DATA_FILE = DATA_DIR / "processed_nsl_kdd.joblib"


def load_processed_data() -> dict:
    """Charge les données déjà préparées par preprocess.py."""
    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PROCESSED_DATA_FILE}. "
            "Exécutez d'abord : python src/preprocess.py"
        )

    return joblib.load(PROCESSED_DATA_FILE)


def build_models() -> dict:
    """Crée les quatre modèles demandés pour comparer plusieurs approches ML."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
    }


def train_and_save_models(models: dict, x_train, y_train) -> None:
    """Entraîne chaque modèle et le sauvegarde dans le dossier models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print(f"Entraînement du modèle : {model_name}")
        model.fit(x_train, y_train)

        output_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(model, output_path)
        print(f"Modèle sauvegardé : {output_path}")


def main() -> None:
    """Point d'entrée principal du script d'entraînement."""
    data = load_processed_data()
    models = build_models()
    train_and_save_models(models, data["X_train"], data["y_train"])
    print("Tous les modèles ont été entraînés avec succès.")


if __name__ == "__main__":
    main()
