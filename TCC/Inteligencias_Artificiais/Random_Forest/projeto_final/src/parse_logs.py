import json
import os
import re
import joblib
import pandas as pd
from src.feature_extractor import (extract_features, xss_indicator_score)


BODY_PATTERN = re.compile(r"\bBODY:\s*(.*)$", re.IGNORECASE)
IP_PATTERN = re.compile(r"\bIP:\s*([0-9a-fA-F:.]+)")
TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))")

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
    timestamp_match = TIMESTAMP_PATTERN.search(linha)
    ip = ip_match.group(1) if ip_match else ""
    body = body_match.group(1).strip() if body_match else linha
    timestamp = timestamp_match.group(1) if timestamp_match else ""
    texto_analisado = body

    try:
        body_json = json.loads(body)
        valores = _extrair_strings_json(body_json)
        if valores:
            texto_analisado = " ".join(valores)
    except json.JSONDecodeError:
        pass

    return {
        "Linha_Log": linha,
        "IP": ip,
        "Body": body,
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


def processar_logs_reais(caminho_logs, caminho_saida, threshold=0.80, regex_threshold=7):

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
    probs = model.predict_proba(X)[:, 1]
    pred_modelo = model.predict(X)
    resultados = []

    for registro, features, prob, pred in zip(registros, features_batch, probs, pred_modelo):
        score_regex = xss_indicator_score(features)
        regex_detectou = bool(score_regex >= regex_threshold)
        rf_detectou = bool(prob >= threshold and score_regex >= 3)
        ataque = rf_detectou or regex_detectou

        if rf_detectou and regex_detectou:
            origem = "(Regex_Redundante)"
        elif rf_detectou:
            origem = "(Random_Forest)"
        elif regex_detectou:
            origem = "(Filtro_Regex)"
        else:
            origem = "(Normal)"

        resultados.append({
            "Data/Hora": registro["Data_Hora"],
            "IP": registro["IP"],
            "Payload": registro["Body"],
            "Indicadores_XSS": int(score_regex),
            "Classificacao": "ATAQUE" if ataque else "NORMAL",
            "Probabilidade_Ataque_IA": f"{round(float(prob) * 100, 1)}%",
            "Tipo_Deteccao": origem,
            "_Prob_RF_Fid": round(float(prob) * 100, 4) 
        })

    df_resultado = pd.DataFrame(resultados)

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df_salvar = df_resultado.drop(columns=["_Prob_RF_Fid"])
    df_salvar.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    ataques = int((df_resultado["Classificacao"] == "ATAQUE").sum())
    normais = int((df_resultado["Classificacao"] == "NORMAL").sum())

    print("\n===== RESUMO =====")
    print(f"Total analisado : {len(df_resultado)}")
    print(f"Ataques         : {ataques}")
    print(f"Normais         : {normais}")
    print(f"\n[SUCESSO] Relatorio salvo em: {caminho_saida}")

    print("\n===== PROBABILIDADES RF =====")
    print("Minima :", round(df_resultado["_Prob_RF_Fid"].min(), 2), "%")
    print("Media  :", round(df_resultado["_Prob_RF_Fid"].mean(), 2), "%")
    print("Maxima :", round(df_resultado["_Prob_RF_Fid"].max(), 2), "%")

    return df_resultado