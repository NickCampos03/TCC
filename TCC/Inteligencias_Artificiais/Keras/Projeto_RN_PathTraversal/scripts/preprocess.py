import os
import random
import urllib.parse

import pandas as pd

try:
    from scripts.text_features import enriquecer_payload
    from scripts.text_features import limpar_e_normalizar_texto
    from scripts.text_features import parece_sqli
except ModuleNotFoundError:
    from text_features import enriquecer_payload
    from text_features import limpar_e_normalizar_texto
    from text_features import parece_sqli


SQLI_RAW_PATH = 'data/SQLi/sqli_raw.csv'
SQLI_3MI_PATH = 'data/SQLi/SQLi_dataset_3Mi.csv'
XSS_RAW_PATH = 'data/XSS/xss_raw.csv'
PATH_TRAVERSAL_RAW_PATH = os.getenv(
    'PATH_TRAVERSAL_DATASET',
    'data/Path_Traversal/path_traversal_complex_50k.csv',
)
OUTPUT_PATH = 'data/dataset_hibrido.csv'

RANDOM_STATE = 42
CHUNK_SIZE = int(os.getenv('PREPROCESS_CHUNK_SIZE', '50000'))
MAX_SQLI_3MI_ROWS = int(os.getenv('MAX_SQLI_3MI_ROWS', '300000'))


def limpar_e_normalizar(payload):
    """Decodifica e normaliza strings para reduzir bypasses simples."""
    return limpar_e_normalizar_texto(payload)


def normalizar_label(valor):
    """Converte formatos comuns de label para 0/1."""
    if pd.isna(valor):
        return None

    texto = str(valor).strip().lower()
    if texto in {'path traversal', 'path_traversal', 'pathtraversal', 'directory traversal', 'directory_traversal', 'traversal'}:
        return 3
    if texto in {'1', 'attack', 'ataque', 'malicious', 'sqli', 'xss', 'true', 'yes'}:
        return 1
    if texto in {'0', 'normal', 'benign', 'legit', 'false', 'no'}:
        return 0

    try:
        return int(float(texto))
    except Exception:
        return None


def detectar_colunas(df):
    """Encontra as colunas de texto e label nos formatos conhecidos."""
    df.columns = [str(c).strip().lower() for c in df.columns]

    candidatos_texto = [
        'payload',
        'query',
        'full_query',
        'text',
        'sentence',
        'request',
        'user_inputs',
    ]
    candidatos_label = ['label', 'class', 'classification', 'target']

    coluna_texto = next((c for c in candidatos_texto if c in df.columns), df.columns[0])
    coluna_label = next((c for c in candidatos_label if c in df.columns), df.columns[-1])
    return coluna_texto, coluna_label


def preparar_dataframe_textual(df, classe_ataque=None):
    coluna_texto, coluna_label = detectar_colunas(df)
    saida = pd.DataFrame({
        'payload': df[coluna_texto].map(enriquecer_payload),
        'label': df[coluna_label].map(normalizar_label),
    })
    if classe_ataque is not None:
        saida['class_label'] = saida['label'].map(lambda label: classe_ataque if label == 1 else 0)
    return saida.dropna()


def carregar_sqli_csv(path):
    if not os.path.exists(path):
        print(f"Aviso: arquivo {path} nao encontrado. Pulando.")
        return pd.DataFrame(columns=['payload', 'label'])

    df = pd.read_csv(path)
    print(f"Processando {len(df)} linhas de SQLi em {path}...")
    df_preparado = preparar_dataframe_textual(df, classe_ataque=2)
    antes = len(df_preparado)
    df_preparado = df_preparado[(df_preparado['label'] == 0) | (df_preparado['payload'].map(parece_sqli))]
    removidos = antes - len(df_preparado)
    if removidos:
        print(f"Aviso: removendo {removidos} registros SQLi sem sinal textual de injecao em {path}.")
    return df_preparado


def carregar_sqli_3mi(path, limite_total=MAX_SQLI_3MI_ROWS):
    """Le o dataset grande em chunks e cria uma amostra balanceada por classe."""
    if not os.path.exists(path):
        print(f"Aviso: dataset 3Mi nao encontrado em {path}.")
        return pd.DataFrame(columns=['payload', 'label'])

    limite_total = max(0, int(limite_total))
    if limite_total == 0:
        print("Aviso: MAX_SQLI_3MI_ROWS=0. Dataset 3Mi ignorado.")
        return pd.DataFrame(columns=['payload', 'label'])

    limite_por_classe = max(1, limite_total // 2)
    rng = random.Random(RANDOM_STATE)
    reservatorios = {0: [], 1: []}
    vistos = {0: 0, 1: 0}

    print(
        f"Lendo SQLi 3Mi em chunks ({path}) com limite de "
        f"{limite_total} registros balanceados..."
    )

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        coluna_texto, coluna_label = detectar_colunas(chunk)
        coluna_inputs = 'user_inputs' if 'user_inputs' in chunk.columns else None
        coluna_status = 'attack_status' if 'attack_status' in chunk.columns else None

        colunas_iteracao = [coluna_texto, coluna_label]
        if coluna_inputs:
            colunas_iteracao.append(coluna_inputs)
        if coluna_status:
            colunas_iteracao.append(coluna_status)

        for row in chunk[colunas_iteracao].itertuples(index=False, name=None):
            texto = row[0]
            label_bruto = row[1]
            user_inputs = row[2] if coluna_inputs else ''
            attack_status = row[3] if coluna_inputs and coluna_status else (row[2] if coluna_status else '')
            label = normalizar_label(label_bruto)
            if label not in reservatorios or pd.isna(texto):
                continue

            # Ataques marcados como "failure" frequentemente deixam uma consulta
            # visualmente normal. Treinar esses casos como SQLi injeta ruido no modelo.
            if label == 1 and coluna_status and str(attack_status).strip().lower() != 'success':
                continue

            texto_treino = f"{texto} {user_inputs}".strip() if label == 1 and coluna_inputs else texto
            if label == 1 and not parece_sqli(texto_treino):
                continue

            vistos[label] += 1
            item = (texto_treino, label)

            if len(reservatorios[label]) < limite_por_classe:
                reservatorios[label].append(item)
                continue

            posicao = rng.randint(0, vistos[label] - 1)
            if posicao < limite_por_classe:
                reservatorios[label][posicao] = item

    registros = reservatorios[0] + reservatorios[1]
    rng.shuffle(registros)

    df = pd.DataFrame(registros, columns=['payload', 'label'])
    if df.empty:
        print("Aviso: nenhum registro valido encontrado no dataset SQLi 3Mi.")
        return df

    df['payload'] = df['payload'].map(enriquecer_payload)
    df['class_label'] = df['label'].map(lambda label: 2 if label == 1 else 0)
    print(
        "SQLi 3Mi selecionado: "
        f"{len(df)} registros | distribuicao {df['label'].value_counts().to_dict()}"
    )
    return df.dropna()


def carregar_xss(path):
    if not os.path.exists(path):
        print(f"Aviso: XSS nao encontrado em {path}.")
        return pd.DataFrame(columns=['payload', 'label'])

    print("Unificando com o dataset de XSS...")
    df_xss_raw = pd.read_csv(path)
    df_xss_raw.columns = df_xss_raw.columns.str.lower()

    if 'payload' in df_xss_raw.columns or 'text' in df_xss_raw.columns:
        return preparar_dataframe_textual(df_xss_raw, classe_ataque=1)

    print("Aviso: convertendo colunas numericas de XSS em representacao textual para o TF-IDF...")
    col_lbl = 'class' if 'class' in df_xss_raw.columns else ('label' if 'label' in df_xss_raw.columns else df_xss_raw.columns[-1])
    colunas_features = [c for c in df_xss_raw.columns if c != col_lbl]
    registros = []

    for row in df_xss_raw.itertuples(index=False):
        dados = dict(zip(df_xss_raw.columns, row))
        tags_ativas = []
        marcadores_xss = []
        for coluna in colunas_features:
            valor = dados[coluna]
            if valor <= 0:
                continue

            nome = coluna.replace('_', ' ')
            if coluna.endswith('length') or coluna in {'url_length', 'url_special_characters'}:
                tags_ativas.append(f"{nome} bucket {min(int(valor) // 10, 50)}")
            else:
                tags_ativas.append(f"{nome} count {min(int(valor), 20)}")

            # Mantém sinais de XSS alinhados aos marcadores extraídos dos logs HTTP.
            if coluna == 'html_tag_script':
                marcadores_xss.append('__xss_script__')
            if coluna.startswith('html_event_'):
                marcadores_xss.append('__xss_event__')
            if coluna == 'js_pseudo_protocol':
                marcadores_xss.append('__xss_js_uri__')
            if coluna.startswith(('js_dom_', 'js_prop_')):
                marcadores_xss.append('__xss_dom__')
            if coluna in {'html_tag_img', 'html_tag_svg', 'html_tag_iframe', 'html_tag_object', 'html_tag_embed', 'html_tag_form', 'html_tag_input'}:
                marcadores_xss.append('__xss_tag__')

        texto_features = enriquecer_payload(" ".join(tags_ativas))
        if marcadores_xss:
            texto_features = f"{texto_features} {' '.join(sorted(set(marcadores_xss)))}"

        registros.append({
            'payload': texto_features,
            'label': normalizar_label(dados[col_lbl]),
        })

    df = pd.DataFrame(registros).dropna()
    df['class_label'] = df['label'].map(lambda label: 1 if label == 1 else 0)
    return df


def carregar_path_traversal(path):
    """Carrega payloads rotulados como Path Traversal (classe 3) e benignos."""
    if not os.path.exists(path):
        print(f"Aviso: dataset de Path Traversal nao encontrado em {path}.")
        return pd.DataFrame(columns=['payload', 'label', 'class_label'])

    df = pd.read_csv(path)
    coluna_texto, coluna_label = detectar_colunas(df)
    labels = df[coluna_label].map(normalizar_label)
    # A coluna pode vir binária (0/1) ou já conter labels de quatro classes.
    class_labels = labels.map(lambda valor: 0 if valor == 0 else (3 if valor in (1, 3) else None))
    payloads = df[coluna_texto].map(enriquecer_payload)
    payloads = [
        f'{payload} __path_traversal_attack__' if classe == 3 else payload
        for payload, classe in zip(payloads, class_labels)
    ]
    saida = pd.DataFrame({
        'payload': payloads,
        'label': class_labels,
        'class_label': class_labels,
    }).dropna()
    saida['label'] = saida['label'].astype(int)
    saida['class_label'] = saida['class_label'].astype(int)
    print(f"Path Traversal carregado: {len(saida)} registros de {path}.")
    return saida


def remover_payloads_conflitantes(df):
    coluna_alvo = 'class_label' if 'class_label' in df.columns else 'label'
    conflitos = df.groupby('payload')[coluna_alvo].nunique()
    payloads_conflitantes = conflitos[conflitos > 1].index

    if len(payloads_conflitantes) == 0:
        return df

    print(f"Aviso: removendo {len(payloads_conflitantes)} payloads com labels conflitantes.")
    return df[~df['payload'].isin(payloads_conflitantes)]


def processar_e_unificar():
    print("--- Iniciando Processamento Baseado em Texto (NLP) ---")

    df_sqli_final = pd.concat(
        [
            carregar_sqli_csv(SQLI_RAW_PATH),
            carregar_sqli_3mi(SQLI_3MI_PATH),
        ],
        ignore_index=True,
    )
    df_xss_final = carregar_xss(XSS_RAW_PATH)
    df_path_final = carregar_path_traversal(PATH_TRAVERSAL_RAW_PATH)

    df_completo = pd.concat([df_xss_final, df_sqli_final, df_path_final], ignore_index=True)
    if df_completo.empty:
        print("Erro: nenhum dataset valido foi encontrado para gerar o treino.")
        return

    df_completo['label'] = df_completo['label'].astype(int)
    if 'class_label' in df_completo.columns:
        df_completo['class_label'] = df_completo['class_label'].astype(int)
    df_completo = remover_payloads_conflitantes(df_completo)
    df_completo = (
        df_completo
        .dropna()
        .drop_duplicates()
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_completo.to_csv(OUTPUT_PATH, index=False)
    print(f"Sucesso! Dataset hibrido de texto salvo em: {OUTPUT_PATH}")
    print(f"Total de registros: {len(df_completo)}")
    print(f"Distribuicao das classes: {df_completo['label'].value_counts().to_dict()}")
    if 'class_label' in df_completo.columns:
        nomes = {0: 'Benigno', 1: 'XSS', 2: 'SQLi', 3: 'Path Traversal'}
        distribuicao_multiclasse = {
            nomes.get(k, str(k)): int(v)
            for k, v in df_completo['class_label'].value_counts().sort_index().items()
        }
        print(f"Distribuicao multiclasse: {distribuicao_multiclasse}")


if __name__ == "__main__":
    processar_e_unificar()
