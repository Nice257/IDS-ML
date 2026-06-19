from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILES = [
    PROJECT_ROOT / "data" / "KDDTest+",
    PROJECT_ROOT / "data" / "KDDTest+.txt",
]
OUTPUT_FILE = PROJECT_ROOT / "data" / "KDDTest_clean.csv"

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


def main() -> None:
    input_file = next((file_path for file_path in INPUT_FILES if file_path.exists()), None)

    if input_file is None:
        expected_files = ", ".join(str(file_path) for file_path in INPUT_FILES)
        raise FileNotFoundError(f"Fichier introuvable. Fichiers attendus : {expected_files}")

    df = pd.read_csv(input_file, names=NSL_KDD_COLUMNS)
    df = df.drop(columns=["difficulty"])
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Conversion terminée : {len(df)} lignes converties")
    print(f"Fichier sauvegardé : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
