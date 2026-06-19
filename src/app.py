"""
Interface Streamlit pour prédire les intrusions réseau avec les modèles IDS-ML.
"""

from pathlib import Path

import joblib
import pandas as pd
import seaborn as sns
import streamlit as st

from preprocess import NSL_KDD_COLUMNS, prepare_input_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"

MODEL_FILES = {
    "Random Forest": MODELS_DIR / "random_forest.pkl",
    "Decision Tree": MODELS_DIR / "decision_tree.pkl",
    "SVM": MODELS_DIR / "svm.pkl",
    "KNN": MODELS_DIR / "knn.pkl",
}

LABELS = {0: "Normal", 1: "Attaque"}


@st.cache_resource
def load_model(model_path: Path):
    """Charge un modèle sauvegardé avec joblib."""
    return joblib.load(model_path)


@st.cache_resource
def load_preprocessor() -> dict:
    """Charge les encodeurs et le scaler sauvegardés par preprocess.py."""
    return joblib.load(PREPROCESSOR_FILE)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Lit le CSV uploadé et ajoute les colonnes NSL-KDD si elles sont absentes."""
    df = pd.read_csv(uploaded_file)

    if not set(["protocol_type", "service", "flag"]).issubset(df.columns):
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, names=NSL_KDD_COLUMNS)

    return df


def predict_traffic(df: pd.DataFrame, model, preprocessor: dict) -> pd.DataFrame:
    """Prépare les données uploadées et ajoute les prédictions au tableau."""
    x_input = prepare_input_features(df, preprocessor)
    predictions = model.predict(x_input)

    result_df = df.copy()
    result_df["prediction"] = predictions
    result_df["prediction_label"] = result_df["prediction"].map(LABELS)
    return result_df


def display_summary(result_df: pd.DataFrame) -> None:
    """Affiche les indicateurs clés et les graphiques de résultats."""
    prediction_counts = result_df["prediction_label"].value_counts()
    normal_count = int(prediction_counts.get("Normal", 0))
    attack_count = int(prediction_counts.get("Attaque", 0))
    total_count = len(result_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total_count)
    col2.metric("Normal", normal_count)
    col3.metric("Attaque", attack_count)

    st.subheader("Distribution des prédictions")
    st.bar_chart(prediction_counts)

    if "protocol_type" in result_df.columns:
        st.subheader("Prédictions par protocole")
        protocol_summary = pd.crosstab(
            result_df["protocol_type"],
            result_df["prediction_label"],
        )
        st.bar_chart(protocol_summary)

    st.subheader("Aperçu des prédictions")
    st.dataframe(result_df, use_container_width=True)


def configure_page() -> None:
    """Configure la page Streamlit et son style de base."""
    st.set_page_config(
        page_title="IDS-ML",
        layout="wide",
    )
    sns.set_theme(style="whitegrid")


def main() -> None:
    """Point d'entrée principal de l'application Streamlit."""
    configure_page()

    st.title("IDS-ML")
    st.caption("Détection d'intrusions réseau avec Machine Learning sur NSL-KDD")

    available_models = {
        model_name: model_path
        for model_name, model_path in MODEL_FILES.items()
        if model_path.exists()
    }

    if not PREPROCESSOR_FILE.exists() or not available_models:
        st.error(
            "Préprocesseur ou modèles introuvables. "
            "Exécutez d'abord `python src/preprocess.py`, puis `python src/train.py`."
        )
        st.stop()

    selected_model = st.sidebar.selectbox("Modèle", list(available_models.keys()))
    uploaded_file = st.sidebar.file_uploader("Fichier CSV de trafic réseau", type=["csv"])

    if uploaded_file is None:
        st.info("Importez un fichier CSV NSL-KDD ou un CSV avec les colonnes de trafic.")
        st.stop()

    model = load_model(available_models[selected_model])
    preprocessor = load_preprocessor()

    try:
        traffic_df = read_uploaded_csv(uploaded_file)
        result_df = predict_traffic(traffic_df, model, preprocessor)
    except Exception as exc:
        st.error(f"Impossible de traiter le fichier : {exc}")
        st.stop()

    st.success(f"Prédictions générées avec le modèle : {selected_model}")
    display_summary(result_df)


if __name__ == "__main__":
    main()
