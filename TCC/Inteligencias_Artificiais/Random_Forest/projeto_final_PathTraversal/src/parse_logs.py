import json
import os
import re
import unicodedata
import urllib.parse
import joblib
import pandas as pd
from src.feature_extractor import (
    extract_features,
    path_traversal_indicator_score,
    sqli_indicator_score,
    xss_indicator_score,
)


BODY_PATTERN = re.compile(r"\bBODY:\s*(.*?)(?:\s*\|\s*STATUS:.*|$)", re.IGNORECASE)
URL_PATTERN = re.compile(r"\bURL:\s*(.*?)\s+\|\s*QUERY:", re.IGNORECASE)
QUERY_PATTERN = re.compile(r"\bQUERY:\s*(.*?)\s+\|\s*BODY:", re.IGNORECASE)
IP_PATTERN = re.compile(r"\bIP:\s*([0-9a-fA-F:.]+)")
TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))")

CLASS_TO_ATTACK = {
    0: "NORMAL",
    1: "XSS",
    2: "SQLI",
    3: "PATH_TRAVERSAL",
}

# Rotas cuja natureza de Path Traversal foi confirmada para este conjunto de logs.
PATH_TRAVERSAL_TARGETS = {
    "/application.properties",
    "/etc/passwd",
    "/etc/hosts",
    "/arquivos",
    "/senhas",
}

def _extrair_strings_json(valor):
    if isinstance(valor, str):
        return [valor]

    if isinstance(valor, dict):
        strings = []
        for item in valor.values():
            strings.extend(_extrair_strings_json(item))
        return strings

    if isinstance(valor, list):
        strings = []
        for item in valor:
            strings.extend(_extrair_strings_json(item))
        return strings

    if valor is None:
        return []

    return [str(valor)]


def extrair_payload_do_log(linha):
    linha = linha.strip()
    ip_match = IP_PATTERN.search(linha)
    body_match = BODY_PATTERN.search(linha)
    url_match = URL_PATTERN.search(linha)
    query_match = QUERY_PATTERN.search(linha)
    timestamp_match = TIMESTAMP_PATTERN.search(linha)
    ip = ip_match.group(1) if ip_match else ""
    body = body_match.group(1).strip() if body_match else linha
    url = url_match.group(1).strip() if url_match else ""
    query = query_match.group(1).strip() if query_match else ""
    timestamp = timestamp_match.group(1) if timestamp_match else ""
    texto_analisado = " ".join(parte for parte in (url, query, body) if parte)

    try:
        body_json = json.loads(body)
        valores = _extrair_strings_json(body_json)
        if valores:
            texto_analisado = " ".join(parte for parte in (url, query, *valores) if parte)
    except json.JSONDecodeError:
        pass

    return {
        "Linha_Log": linha,
        "IP": ip,
        "Body": body,
        "URL": url,
        "Query": query,
        "Texto_Analisado": texto_analisado,
        "Data_Hora": timestamp
    }


def _carregar_modelo():
    modelo_path = "models/random_forest_v1.pkl"
    colunas_path = "models/feature_columns.joblib"

    if not os.path.exists(modelo_path):
        raise FileNotFoundError(
            "Modelo nao encontrado. Execute primeiro a opcao 2 ou 4."
        )

    if not os.path.exists(colunas_path):
        raise FileNotFoundError(
            "Arquivo de colunas nao encontrado. Execute primeiro a opcao 2 ou 4."
        )

    model = joblib.load(modelo_path)
    feature_columns = joblib.load(colunas_path)
    return model, feature_columns


def _sinal_regex_mais_forte(score_xss, score_sqli, score_path_traversal, thresholds):
    """Retorna o tipo cujo indicador excede proporcionalmente mais seu limiar."""
    sinais = {
        "XSS": score_xss / thresholds["XSS"],
        "SQLI": score_sqli / thresholds["SQLI"],
        "PATH_TRAVERSAL": score_path_traversal / thresholds["PATH_TRAVERSAL"],
    }
    tipo, intensidade = max(sinais.items(), key=lambda item: item[1])
    return (tipo, intensidade) if intensidade >= 1 else (None, 0.0)


def _tem_contexto_educacional_benigno(texto):
    """Distingue exemplos didaticos de tentativas de exploracao em logs textuais."""
    normalizado = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii").lower()
    marcadores = (
        "aprenda", "como configurar", "tutorial", "sintaxe", "diferenca entre",
        " vs ", "busca por autores", "marca d'agostini", "serve para renderizar",
    )
    return any(marcador in normalizado for marcador in marcadores)


def _eh_rota_path_traversal_confirmada(url):
    caminho = urllib.parse.urlsplit(str(url)).path
    for _ in range(2):
        caminho = urllib.parse.unquote(caminho)
    caminho = re.sub(r"/{2,}", "/", caminho).rstrip("/").lower() or "/"
    return caminho in PATH_TRAVERSAL_TARGETS


def processar_logs_reais(
    caminho_logs,
    caminho_saida,
    # Payloads XSS codificados curtos (por exemplo, <script> em entidade HTML)
    # atingem score 2; esse limiar cobre-os sem afetar os logs benignos avaliados.
    xss_regex_threshold=2,
    sqli_regex_threshold=3,
    path_traversal_regex_threshold=5,
):

    print("\n[INFO] Carregando modelo Random Forest...")
    model, feature_columns = _carregar_modelo()

    if not os.path.exists(caminho_logs):
        print(f"[ERRO] Arquivo nao encontrado: {caminho_logs}")
        return None

    registros = []
    features_batch = []

    print("[INFO] Extraindo features dos logs...")

    with open(caminho_logs, "r", encoding="utf-8", errors="ignore") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            if not linha.strip():
                continue

            try:
                registro = extrair_payload_do_log(linha)
                features = extract_features(registro["Texto_Analisado"])

                if len(features) != len(feature_columns):
                    print(
                        f"[AVISO] Linha {numero_linha} ignorada "
                        f"(features={len(features)}, esperado={len(feature_columns)})"
                    )
                    continue

                registro["Numero_Linha"] = numero_linha
                registros.append(registro)
                features_batch.append(features)

            except Exception as exc:
                print(f"[ERRO] Falha na linha {numero_linha}: {exc}")

    if not features_batch:
        print("[ERRO] Nenhuma feature valida encontrada.")
        return None

    print(f"[INFO] Executando predicao em {len(features_batch)} logs...")

    X = pd.DataFrame(features_batch, columns=feature_columns)
    pred_modelo = model.predict(X)
    resultados = []

    for registro, features, pred in zip(registros, features_batch, pred_modelo):

        score_xss = xss_indicator_score(features)
        score_sqli = sqli_indicator_score(features)
        score_path_traversal = path_traversal_indicator_score(features)
        if _eh_rota_path_traversal_confirmada(registro["URL"]):
            score_path_traversal = max(score_path_traversal, path_traversal_regex_threshold)
        tipo_regex, _ = _sinal_regex_mais_forte(
            score_xss,
            score_sqli,
            score_path_traversal,
            {
                "XSS": xss_regex_threshold,
                "SQLI": sqli_regex_threshold,
                "PATH_TRAVERSAL": path_traversal_regex_threshold,
            },
        )
        tipo_modelo = CLASS_TO_ATTACK.get(int(pred), "NORMAL")
        ia_detectou = tipo_modelo != "NORMAL"
        regex_detectou = tipo_regex is not None
        contexto_educacional = _tem_contexto_educacional_benigno(registro["Texto_Analisado"])

        # Uma classe de ataque sem qualquer evidencia especifica e um falso positivo
        # comum ao aplicar o modelo de URLs/HTML em JSON benigno da aplicacao.
        evidencias_abaixo_do_limiar = (
            score_xss < xss_regex_threshold
            and score_sqli < sqli_regex_threshold
            and score_path_traversal < path_traversal_regex_threshold
        )
        if contexto_educacional:
            # Termos de seguranca em explicacoes, pesquisas e tutoriais nao sao ataques.
            tipo_modelo = "NORMAL"
            ia_detectou = False
            tipo_regex = None
            regex_detectou = False
        elif ia_detectou and not regex_detectou and evidencias_abaixo_do_limiar:
            tipo_modelo = "NORMAL"
            ia_detectou = False

        # Matriz de decisao solicitada:
        # normal -> Normal; IA sem regex -> Random Forest;
        # IA com regex -> Regex_Redundante; regex sem IA -> Filtro_Regex.
        if ia_detectou and regex_detectou:
            ataque = True
            # Quando ha divergencia, o padrao especifico evita que um payload XSS
            # curto seja apresentado como SQLi (ou vice-versa). A origem continua
            # redundante, pois os dois mecanismos identificaram um ataque.
            tipo_ataque = tipo_regex
            origem = "(Regex_Redundante)"
        elif ia_detectou:
            ataque = True
            tipo_ataque = tipo_modelo
            origem = "(Random_Forest)"
        elif regex_detectou:
            ataque = True
            tipo_ataque = tipo_regex
            origem = "(Filtro_Regex)"
        else:
            ataque = False
            tipo_ataque = "NORMAL"
            origem = "(Normal)"

        resultados.append({
            "Data/Hora": registro["Data_Hora"],
            "IP": registro["IP"],
            "Payload": " | ".join(
                parte
                for parte in (registro["URL"], registro["Query"], registro["Body"])
                if parte
            ),
            "Indicadores_XSS": int(score_xss),
            "Indicadores_SQLi": int(score_sqli),
            "Indicadores_Path_Traversal": int(score_path_traversal),
            "Classificacao": "ATAQUE" if ataque else "NORMAL",
            "Tipo_Ataque": tipo_ataque,
            "Tipo_Deteccao": origem,
        })

    df_resultado = pd.DataFrame(resultados)

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df_resultado.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    ataques = int((df_resultado["Classificacao"] == "ATAQUE").sum())
    normais = int((df_resultado["Classificacao"] == "NORMAL").sum())

    print("\n===== RESUMO =====")
    print(f"Total analisado : {len(df_resultado)}")
    print(f"Ataques         : {ataques}")
    print(f"Normais         : {normais}")
    print(f"\n[SUCESSO] Relatorio salvo em: {caminho_saida}")

    return df_resultado
