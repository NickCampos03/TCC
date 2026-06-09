import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.feature_extractor import extract_features, xss_indicator_score

def analisar_logs_neural_network():
    print("\n--- [PASSO 3: ANALISANDO LOGS COM NORMALIZACAO CORRETA] ---")

    log_path = 'data/app_logs.log.txt'
    model_path = 'models/model_nn.keras'
    scaler_path = 'models/scaler_xss.joblib'
    features_path = 'models/feature_columns.joblib'
    output_report = 'reports/resultado_analise.csv'

    if not os.path.exists(log_path):
        print("Erro: Arquivo de log nao encontrado.")
        return

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
        print("Erro: Modelo, scaler ou lista de features nao encontrados. Treine a rede primeiro.")
        return

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(features_path)

    processed_rows = []
    log_pattern = re.compile(r"^([^\s,]+)\s+INFO\s+IP:\s+([^\s]+)\s+BODY:\s+(.*)$", re.IGNORECASE)

    with open(log_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = log_pattern.match(line.strip())
            if not match:
                continue

            payload = match.group(3)
            feature_values = extract_features(payload)
            indicator_score = xss_indicator_score(payload)
            if len(feature_values) != len(feature_columns):
                raise ValueError(
                    f"Payload gerou {len(feature_values)} features, mas o modelo espera {len(feature_columns)}."
                )

            processed_rows.append({
                'Data/Hora': match.group(1),
                'IP': match.group(2),
                'Payload': payload,
                'Features': feature_values,
                'Indicadores_XSS': indicator_score,
            })

    if not processed_rows:
        print("Nenhum log valido para analisar.")
        return

    df_logs = pd.DataFrame(processed_rows)
    X = np.array(df_logs['Features'].tolist(), dtype=np.float32)
    X_scaled = scaler.transform(X)

    predictions_prob = model.predict(X_scaled, verbose=0).flatten()

    def classify(ai_detected, regex_detected):
        if ai_detected or regex_detected:
            return "ATAQUE"
        return "NORMAL"

    def detection_type(ai_detected, regex_detected):
        if ai_detected and regex_detected:
            return "Regex_Redundante"
        if ai_detected:
            return "Deteccao_IA"
        if regex_detected:
            return "Filtro_Regex"
        return "Nao_Detectado"

    ai_probabilities = []
    classifications = []
    detection_types = []
    for probability, indicator_score in zip(predictions_prob, df_logs['Indicadores_XSS']):
        effective_ai_probability = float(probability) if indicator_score > 0 else 0.0
        ai_detected = effective_ai_probability > 0.7
        regex_detected = indicator_score >= 2

        classifications.append(classify(ai_detected, regex_detected))
        detection_types.append(f"({detection_type(ai_detected, regex_detected)})")
        ai_probabilities.append(effective_ai_probability)

    df_logs['Classificacao'] = classifications
    df_logs['Tipo_Deteccao'] = detection_types
    df_logs['Probabilidade_Ataque_IA'] = [f"{probability * 100:.1f}%" for probability in ai_probabilities]

    os.makedirs('reports', exist_ok=True)
    report_columns = [
        'Data/Hora',
        'IP',
        'Payload',
        'Indicadores_XSS',
        'Classificacao',
        'Probabilidade_Ataque_IA',
        'Tipo_Deteccao',
    ]
    df_logs.drop(columns=['Features'])[report_columns].to_csv(output_report, index=False, encoding='utf-8-sig')
    print(f"Analise concluida. Relatorio salvo em: {output_report}")


if __name__ == "__main__":
    analisar_logs_neural_network()
