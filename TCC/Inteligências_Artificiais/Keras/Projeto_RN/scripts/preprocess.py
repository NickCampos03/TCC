import os
import pandas as pd
from scripts.feature_extractor import FEATURE_COLUMNS

XSS_RAW_PATH = 'data/xss_raw.csv'
OUTPUT_PATH = 'data/dataset_xss.csv'
LABEL_COLUMN = 'label'

def processar_e_unificar():
    print("--- [PASSO 1] Preparando dataset numerico xss_raw ---")
    if not os.path.exists(XSS_RAW_PATH):
        print(f"Erro: {XSS_RAW_PATH} nao encontrado.")
        return

    df = pd.read_csv(XSS_RAW_PATH)
    df.columns = [str(column).strip() for column in df.columns]

    if LABEL_COLUMN not in df.columns:
        df = df.rename(columns={df.columns[-1]: LABEL_COLUMN})

    missing = [column for column in FEATURE_COLUMNS + [LABEL_COLUMN] if column not in df.columns]
    if missing:
        print("Erro: colunas obrigatorias ausentes no dataset:")
        print(", ".join(missing))
        return

    df = df[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    before = len(df)
    df = df.dropna()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df = df[df[LABEL_COLUMN].isin([0, 1])]

    if df.empty:
        print("Erro: dataset ficou vazio apos limpeza numerica.")
        return

    df.to_csv(OUTPUT_PATH, index=False)
    removed = before - len(df)
    print(f"Dataset numerico pronto em: {OUTPUT_PATH} com {len(df)} linhas ({removed} removidas).")
