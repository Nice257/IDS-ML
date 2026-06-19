"""
Téléchargement automatique des fichiers NSL-KDD nécessaires au projet IDS-ML.
"""

from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DATASET_URLS = {
    "KDDTrain+.txt": (
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"
        "KDDTrain%2B.txt"
    ),
    "KDDTest+.txt": (
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"
        "KDDTest%2B.txt"
    ),
}


def download_file(file_name: str, url: str) -> None:
    """Télécharge un fichier NSL-KDD dans le dossier data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / file_name

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Déjà présent : {output_path}")
        return

    print(f"Téléchargement : {file_name}")
    urlretrieve(url, output_path)
    print(f"Sauvegardé : {output_path}")


def main() -> None:
    """Point d'entrée principal du script de téléchargement."""
    for file_name, url in DATASET_URLS.items():
        download_file(file_name, url)

    print("Dataset NSL-KDD prêt dans le dossier data/.")


if __name__ == "__main__":
    main()
