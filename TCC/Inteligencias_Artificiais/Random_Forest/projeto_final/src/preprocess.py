import os
import joblib
import pandas as pd
from src.feature_extractor import FEATURE_COLUMNS


RAW_DATASET_PATH = "data/raw/XSS/XSS_dataset.csv"
PROCESSED_DATASET_PATH = "data/processed/XSS/dataset_distribuido.csv"

# 1. VALIDAÇÃO E FILTRAGEM DE COLUNAS
def _validar_colunas(df):
    expected_columns = FEATURE_COLUMNS + ["Class"]
    missing = [col for col in expected_columns if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_columns]

    if missing:
        raise ValueError("Dataset sem colunas obrigatorias: " + ", ".join(missing))

    if extra:
        df = df.drop(columns=extra)

    return df[expected_columns]


# 2. PIPELINE DE PROCESSAMENTO E LIMPEZA
def processar_e_balancear():
    print("\n[INFO] Carregando dataset bruto...")
    df = pd.read_csv(RAW_DATASET_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = _validar_colunas(df)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df[df["Class"].isin([0, 1])]
    df["Class"] = df["Class"].astype(int)

    linhas_antes = len(df)
    df = df.drop_duplicates()
    duplicadas = linhas_antes - len(df)
    df = df.sample(frac=1, random_state=42)

    os.makedirs("models", exist_ok=True)
    joblib.dump(FEATURE_COLUMNS, "models/feature_columns.joblib")

    os.makedirs(os.path.dirname(PROCESSED_DATASET_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATASET_PATH, index=False)
    distribuicao = df["Class"].value_counts().sort_index()

    print(f"[SUCESSO] Dataset salvo em {PROCESSED_DATASET_PATH}")
    print(f"[INFO] Linhas finais: {len(df)} (duplicadas removidas: {duplicadas})")
    print(f"[INFO] Distribuicao de classes: NORMAL={int(distribuicao.get(0, 0))}, ATAQUE={int(distribuicao.get(1, 0))}")

    return df