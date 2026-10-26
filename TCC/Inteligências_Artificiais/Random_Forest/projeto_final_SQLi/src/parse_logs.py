import re
import pandas as pd
import numpy as np
import joblib
import os
from src.feature_extractor import extrair_features 

def mapear_dinamicamente(dict_extraido, colunas_modelo):
    """Garante que as colunas extraídas batam com as colunas que o modelo espera."""
    input_final = {}
    for col in colunas_modelo:
        valor = 0
        col_lower = col.lower()
        # Procura a chave no dicionário extraído que mais se aproxima da coluna do modelo
        for chave, v in dict_extraido.items():
            if chave == col_lower or chave in col_lower:
                valor = v
                break
        input_final[col] = valor
    return input_final

def processar_logs_reais(arquivo_entrada, arquivo_saida):
    modelo_path = 'models/random_forest_v1.pkl' 
    
    if not os.path.exists(modelo_path):
        print(f"[ERRO] Modelo não encontrado em {modelo_path}")
        return
    
    # 1. Carrega o bundle do modelo
    data_bundle = joblib.load(modelo_path)
    
    # Suporta tanto o dicionário completo quanto apenas o objeto do modelo
    if isinstance(data_bundle, dict):
        clf = data_bundle['model']
        colunas_do_modelo = data_bundle['features']
        # Mapeamento de nomes (ex: {0: 'NORMAL', 1: 'XSS', 2: 'SQLI'})
        target_names = data_bundle.get('target_names', None)
    else:
        clf = data_bundle
        colunas_do_modelo = clf.feature_names_in_
        target_names = None

    # Regex para capturar os campos do seu log
    regex_log = r"(?P<data>\d{4}-\d{2}-\d{2}T[\d:.-]+).*?IP:\s*(?P<ip>[\w\.:]+).*?BODY:\s*(?P<body>\{.*\})"
    dados_analisados = []

    if not os.path.exists(arquivo_entrada):
        print(f"[ERRO] Arquivo de log {arquivo_entrada} não encontrado.")
        return

    print(f"[INFO] Analisando logs e gerando relatório...")

    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        for linha in f:
            match = re.search(regex_log, linha)
            if match:
                body = match.group('body')
                
                # Extração e Mapeamento
                dict_feats = extrair_features(body)
                feats_adaptadas = mapear_dinamicamente(dict_feats, colunas_do_modelo)
                
                # DataFrame de uma linha para o Scikit-Learn
                df_input = pd.DataFrame([feats_adaptadas])[colunas_do_modelo]
                
                # Predição de Classe e Probabilidade
                predicao_raw = clf.predict(df_input)[0]
                probs = clf.predict_proba(df_input)[0]
                confianca = max(probs)

                # Se for número (int), busca o nome na lista de classes. 
                # Se já for string, apenas usa ela.
                if isinstance(predicao_raw, (int, np.integer)):
                    # Caso o target_names exista, usamos ele, senão usamos o clf.classes_
                    lista_classes = target_names if target_names is not None else clf.classes_
                    predicao_nome = str(lista_classes[predicao_raw])
                else:
                    predicao_nome = str(predicao_raw)

                # Agora sim o .upper() funcionará, pois predicao_nome é garantidamente string
                predicao_nome = predicao_nome.upper()

                # --- Lógica de Classificação Final ---
                if predicao_nome == "NORMAL" or predicao_nome == "0": # Tratando caso venha '0'
                    classificacao_final = "NORMAL"
                else:
                    classificacao_final = "ATAQUE"

                # Backup de Segurança (Heurística)
                # Se o modelo marcar NORMAL mas houver algo muito óbvio, forçamos o alerta
                if classificacao_final == "NORMAL" and (dict_feats.get('script', 0) > 0 or dict_feats.get('sqli_com', 0) > 0):
                    classificacao_final = "ATAQUE"
                    predicao_nome = "SENSÍVEL (Heurística)"
                    confianca = 0.5

                dados_analisados.append({
                    'Data/Hora': match.group('data'),
                    'IP': match.group('ip'),
                    'Payload': body,
                    'Classificacao': classificacao_final,
                    'Confianca': f"{confianca * 100:.1f}%",
                    'Suspeita': predicao_nome if classificacao_final == "ATAQUE" else "-"
                })

    # Salva o relatório final
    if dados_analisados:
        df_final = pd.DataFrame(dados_analisados)
        os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)
        df_final.to_csv(arquivo_saida, index=False, encoding='utf-8-sig')
        print(f"[SUCESSO] Relatório salvo em: {arquivo_saida}")
    else:
        print("[AVISO] Nenhum log compatível encontrado para processar.")