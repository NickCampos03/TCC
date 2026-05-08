import re
import pandas as pd
import joblib
import os
from src.feature_extractor import extrair_features 

def mapear_dinamicamente(dict_extraido, colunas_modelo):
    """
    Cruza os dados extraídos com as colunas que o modelo Random Forest espera.
    Isso evita que o modelo ignore dados se os títulos das tabelas forem 
    ligeiramente diferentes dos nomes das chaves no extrator.
    """
    input_final = {}
    for col in colunas_modelo:
        valor = 0
        col_lower = col.lower()
        # Procura correspondência por palavra-chave (ex: 'script' em 'html_tag_script')
        for chave, v in dict_extraido.items():
            if chave in col_lower:
                valor = v
                break
        input_final[col] = valor
    return input_final

def processar_logs_reais(arquivo_entrada, arquivo_saida):
    modelo_path = 'models/random_forest_v1.pkl'
    
    if not os.path.exists(modelo_path):
        print(f"Erro: Modelo não encontrado em {modelo_path}")
        return
    
    # Carrega o modelo e identifica as colunas que ele exige
    modelo = joblib.load(modelo_path)
    colunas_do_modelo = modelo.feature_names_in_

    # Regex para capturar Data, IP e o JSON do BODY
    regex_log = r"(?P<data>\d{4}-\d{2}-\d{2}T[\d:.-]+).*IP:\s*(?P<ip>[a-fA-F\d\.:]+).*BODY:\s*(?P<body>\{.*\})\s*\|\s*STATUS"
    
    dados_analisados = []

    if not os.path.exists(arquivo_entrada):
        print(f"Erro: Arquivo {arquivo_entrada} não encontrado.")
        return

    print(f"Iniciando análise de: {arquivo_entrada}")

    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        for linha in f:
            match = re.search(regex_log, linha)
            if match:
                body = match.group('body')
                
                # 1. Extração baseada em padrões
                dict_feats = extrair_features(body)
                
                # 2. Mapeamento inteligente para as colunas do modelo
                feats_adaptadas = mapear_dinamicamente(dict_feats, colunas_do_modelo)
                
                # 3. Criação do DataFrame respeitando a ordem das colunas do treino
                df_input = pd.DataFrame([feats_adaptadas])[colunas_do_modelo]
                
                # 4. Predição e Probabilidade
                probabilidades = modelo.predict_proba(df_input)[0]
                confianca_bruta = probabilidades[1]

                # --- Lógica de Apoio à Decisão (Guardião) ---
                # O script ajuda o modelo a não ignorar evidências claras de XSS
                evidencia_clara = any([
                    dict_feats.get('script', 0) > 0,
                    dict_feats.get('onerror', 0) > 0,
                    dict_feats.get('javascript_proto', 0) > 0,
                    dict_feats.get('iframe', 0) > 0
                ])

                if evidencia_clara:
                    # Se há evidência, tratamos como ataque mesmo com confiança baixa do modelo
                    predicao_final = 'ATAQUE'
                    # Ajustamos a confiança para refletir a detecção do script
                    exibir_confianca = max(confianca_bruta, 0.85) 
                else:
                    # Se não há evidência clara, seguimos estritamente o modelo
                    predicao = modelo.predict(df_input)[0]
                    predicao_final = 'ATAQUE' if predicao == 1 or confianca_bruta > 0.3 else 'NORMAL'
                    exibir_confianca = confianca_bruta

                dados_analisados.append({
                    'Data/Hora': match.group('data'),
                    'IP': match.group('ip'),
                    'Payload': body,
                    'Classificacao': predicao_final,
                    'Confianca': f"{exibir_confianca * 100:.1f}%"
                })

    if dados_analisados:
        df_final = pd.DataFrame(dados_analisados)
        os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)
        # Salva com utf-8-sig para garantir que o Excel abra os acentos corretamente
        df_final.to_csv(arquivo_saida, index=False, encoding='utf-8-sig')
        print(f"Análise concluída. Relatório salvo em: {arquivo_saida}")
    else:
        print("Nenhum log válido foi encontrado pelo Regex.")

if __name__ == "__main__":
    # Caminhos padrão do seu projeto
    processar_logs_reais('data/app_logs/app_logs.log.txt', 'reports/resultado_analise.csv')