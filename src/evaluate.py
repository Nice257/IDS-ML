"""
Évaluation et comparaison des modèles IDS entraînés sur NSL-KDD.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DATA_FILE = DATA_DIR / "processed_nsl_kdd.joblib"
MODEL_FILES = {
    "Random Forest": MODELS_DIR / "random_forest.pkl",
    "Decision Tree": MODELS_DIR / "decision_tree.pkl",
    "SVM": MODELS_DIR / "svm.pkl",
    "KNN": MODELS_DIR / "knn.pkl",
}


def load_processed_data() -> dict:
    """Charge les données de test produites par preprocess.py."""
    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PROCESSED_DATA_FILE}. "
            "Exécutez d'abord : python src/preprocess.py"
        )

    return joblib.load(PROCESSED_DATA_FILE)


def load_models() -> dict:
    """Charge tous les modèles entraînés disponibles dans models/."""
    models = {}

    for model_name, model_path in MODEL_FILES.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}. "
                "Exécutez d'abord : python src/train.py"
            )
        models[model_name] = joblib.load(model_path)

    return models


def compute_metrics(model, x_test, y_test) -> dict:
    """Calcule les métriques principales d'un modèle de classification binaire."""
    y_pred = model.predict(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Predictions": y_pred,
    }


def plot_confusion_matrix(model_name: str, y_test, y_pred) -> None:
    """Affiche et sauvegarde la matrice de confusion pour un modèle."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attaque"],
        yticklabels=["Normal", "Attaque"],
    )
    plt.title(f"Matrice de confusion - {model_name}")
    plt.xlabel("Prédiction")
    plt.ylabel("Valeur réelle")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{slugify(model_name)}.png", dpi=150)
    plt.close()


def plot_performance_comparison(metrics_df: pd.DataFrame) -> None:
    """Génère un barplot comparatif des performances des modèles."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    long_df = metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=long_df, x="Model", y="Score", hue="Metric")
    plt.ylim(0, 1)
    plt.title("Comparaison des performances des modèles IDS")
    plt.xlabel("Modèle")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_performance_comparison.png", dpi=150)
    plt.close()


def slugify(value: str) -> str:
    """Convertit un nom de modèle en nom de fichier simple."""
    return value.lower().replace(" ", "_").replace("-", "_")


def evaluate_models(models: dict, x_test, y_test) -> pd.DataFrame:
    """Évalue tous les modèles et retourne un tableau de métriques."""
    results = []

    for model_name, model in models.items():
        print(f"Évaluation du modèle : {model_name}", flush=True)
        metrics = compute_metrics(model, x_test, y_test)
        plot_confusion_matrix(model_name, y_test, metrics["Predictions"])

        results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-Score": metrics["F1-Score"],
            }
        )

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    return metrics_df


def main() -> None:
    """Point d'entrée principal du script d'évaluation."""
    data = load_processed_data()
    models = load_models()
    metrics_df = evaluate_models(models, data["X_test"], data["y_test"])

    print(metrics_df.round(4).to_string(index=False))
    plot_performance_comparison(metrics_df)
    print(f"Rapports sauvegardés dans : {REPORTS_DIR}")


if __name__ == "__main__":
    main()
