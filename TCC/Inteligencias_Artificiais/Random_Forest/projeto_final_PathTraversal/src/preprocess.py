import os
import joblib
import pandas as pd
from src.feature_extractor import FEATURE_COLUMNS, extract_features


RAW_XSS_DATASET_PATH = "data/raw/XSS/XSS_dataset.csv"
RAW_SQLI_DATASET_PATHS = [
    # Base curta e rotulada: suficiente para o treino balanceado, sem carregar 3 milhoes de linhas.
    "data/raw/SQLi/SQLi_dataset.csv",
]
# PATH TRAVERSAL DATASET — nova fonte incorporada ao pipeline multiclasse.
RAW_PATH_TRAVERSAL_DATASET_PATH = "data/raw/PathTraversal/path_traversal_complex_50k.csv"
PROCESSED_DATASET_PATH = "data/processed/dataset_multiclasse.csv"
# Cada classe recebe o mesmo limite. Dez mil amostras por classe permitem treino rapido e reproduzivel.
MAX_SAMPLES_PER_CLASS = 10000
SQLI_CHUNKSIZE = 100000

CLASS_NORMAL = 0
CLASS_XSS = 1
CLASS_SQLI = 2
CLASS_PATH_TRAVERSAL = 3

# 1. VALIDAÇÃO E FILTRAGEM DE COLUNAS
def _validar_colunas(df):
    expected_columns = FEATURE_COLUMNS + ["Class"]
    missing = [col for col in expected_columns if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_columns]

    # Inclui todas as features novas de uma vez, evitando fragmentacao do DataFrame.
    missing_features = [col for col in missing if col != "Class"]
    if missing_features:
        df = pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=missing_features)],
            axis=1,
        )

    if "Class" not in df.columns:
        raise ValueError("Dataset sem coluna obrigatoria: Class")

    if extra:
        df = df.drop(columns=extra)

    return df[expected_columns]


def _normalizar_dataset_xss():
    print("[INFO] Carregando dataset XSS...")
    df = pd.read_csv(RAW_XSS_DATASET_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = _validar_colunas(df)
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df[df["Class"].isin([0, 1])]
    df["Class"] = df["Class"].astype(int)
    df.loc[df["Class"] == 1, "Class"] = CLASS_XSS
    return df


def _extrair_amostras_sqli(caminho):
    if not os.path.exists(caminho):
        print(f"[AVISO] Dataset SQLi nao encontrado: {caminho}")
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["Class"])

    print(f"[INFO] Carregando dataset SQLi: {caminho}")
    amostras = []
    contagem = {CLASS_NORMAL: 0, CLASS_SQLI: 0}

    for chunk in pd.read_csv(caminho, chunksize=SQLI_CHUNKSIZE, low_memory=False):
        col_texto = "Query" if "Query" in chunk.columns else "full_query" if "full_query" in chunk.columns else None
        col_label = "Label" if "Label" in chunk.columns else "label" if "label" in chunk.columns else None

        if not col_texto or not col_label:
            raise ValueError(f"Dataset SQLi invalido: {caminho}")

        chunk = chunk[[col_texto, col_label]].dropna(subset=[col_texto, col_label])
        chunk[col_label] = pd.to_numeric(chunk[col_label], errors="coerce")
        chunk = chunk[chunk[col_label].isin([0, 1])]

        for label_origem, classe_destino in [(0, CLASS_NORMAL), (1, CLASS_SQLI)]:
            restante = MAX_SAMPLES_PER_CLASS - contagem[classe_destino]
            if restante <= 0:
                continue

            subset = chunk[chunk[col_label] == label_origem]
            if subset.empty:
                continue

            if len(subset) > restante:
                subset = subset.sample(n=restante, random_state=42)

            features = subset[col_texto].map(extract_features).tolist()
            if features:
                df_features = pd.DataFrame(features, columns=FEATURE_COLUMNS)
                df_features["Class"] = classe_destino
                amostras.append(df_features)
                contagem[classe_destino] += len(df_features)

        if all(valor >= MAX_SAMPLES_PER_CLASS for valor in contagem.values()):
            break

    if not amostras:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["Class"])

    print(f"[INFO] Amostras SQLi extraidas: NORMAL={contagem[CLASS_NORMAL]}, SQLI={contagem[CLASS_SQLI]}")
    return pd.concat(amostras, ignore_index=True)


def _extrair_amostras_path_traversal():
    """PATH TRAVERSAL DATASET: vetorizacao da URL e remapeamento 0/1 para 0/3."""
    if not os.path.exists(RAW_PATH_TRAVERSAL_DATASET_PATH):
        print(f"[AVISO] Dataset Path Traversal nao encontrado: {RAW_PATH_TRAVERSAL_DATASET_PATH}")
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["Class"])

    print("[INFO] Carregando dataset Path Traversal...")
    df = pd.read_csv(RAW_PATH_TRAVERSAL_DATASET_PATH, usecols=["request_url", "label"])
    df = df.dropna(subset=["request_url", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]

    partes = []
    for label_origem, classe_destino in [(0, CLASS_NORMAL), (1, CLASS_PATH_TRAVERSAL)]:
        subset = df[df["label"] == label_origem]
        if len(subset) > MAX_SAMPLES_PER_CLASS:
            subset = subset.sample(n=MAX_SAMPLES_PER_CLASS, random_state=42)
        features = subset["request_url"].map(extract_features).tolist()
        df_features = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        df_features["Class"] = classe_destino
        partes.append(df_features)

    resultado = pd.concat(partes, ignore_index=True)
    print(
        "[INFO] Amostras Path Traversal extraidas: "
        f"NORMAL={int((resultado['Class'] == CLASS_NORMAL).sum())}, "
        f"PATH_TRAVERSAL={int((resultado['Class'] == CLASS_PATH_TRAVERSAL).sum())}"
    )
    return resultado


def _balancear_multiclasse(df):
    contagens = df["Class"].value_counts()
    classes_esperadas = {CLASS_NORMAL, CLASS_XSS, CLASS_SQLI, CLASS_PATH_TRAVERSAL}
    if set(contagens.index) != classes_esperadas:
        raise ValueError("Dataset sem todas as quatro classes necessarias para o treino.")

    # O menor grupo define o teto; nenhuma classe pode sobrepor outra no treinamento.
    amostras_por_classe = min(MAX_SAMPLES_PER_CLASS, int(contagens.min()))
    partes = [
        grupo.sample(n=amostras_por_classe, random_state=42)
        for _, grupo in df.groupby("Class")
    ]
    return pd.concat(partes, ignore_index=True).sample(frac=1, random_state=42)


def _remover_features_com_rotulos_conflitantes(df):
    """Remove vetores identicos usados por mais de uma classe, evitando aprendizado ambiguo."""
    quantidade_rotulos = df.groupby(FEATURE_COLUMNS, dropna=False)["Class"].transform("nunique")
    conflitantes = quantidade_rotulos > 1
    removidas = int(conflitantes.sum())
    return df.loc[~conflitantes].copy(), removidas


# 2. PIPELINE DE PROCESSAMENTO E LIMPEZA
def processar_e_balancear():
    print("\n[INFO] Montando dataset multiclasse (NORMAL, XSS, SQLI, PATH_TRAVERSAL)...")
    datasets = [_normalizar_dataset_xss()]
    datasets.extend(_extrair_amostras_sqli(caminho) for caminho in RAW_SQLI_DATASET_PATHS)
    datasets.append(_extrair_amostras_path_traversal())
    df = pd.concat(datasets, ignore_index=True)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df[df["Class"].isin([CLASS_NORMAL, CLASS_XSS, CLASS_SQLI, CLASS_PATH_TRAVERSAL])]
    df["Class"] = df["Class"].astype(int)

    linhas_antes = len(df)
    df = df.drop_duplicates()
    duplicadas = linhas_antes - len(df)
    df, conflitantes = _remover_features_com_rotulos_conflitantes(df)
    df = _balancear_multiclasse(df)

    os.makedirs("models", exist_ok=True)
    joblib.dump(FEATURE_COLUMNS, "models/feature_columns.joblib")

    os.makedirs(os.path.dirname(PROCESSED_DATASET_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATASET_PATH, index=False)
    distribuicao = df["Class"].value_counts().sort_index()

    print(f"[SUCESSO] Dataset salvo em {PROCESSED_DATASET_PATH}")
    print(
        f"[INFO] Linhas finais: {len(df)} (duplicadas removidas: {duplicadas}; "
        f"vetores conflitantes removidos: {conflitantes})"
    )
    print(
        "[INFO] Distribuicao de classes: "
        f"NORMAL={int(distribuicao.get(CLASS_NORMAL, 0))}, "
        f"XSS={int(distribuicao.get(CLASS_XSS, 0))}, "
        f"SQLI={int(distribuicao.get(CLASS_SQLI, 0))}, "
        f"PATH_TRAVERSAL={int(distribuicao.get(CLASS_PATH_TRAVERSAL, 0))}"
    )

    return df
