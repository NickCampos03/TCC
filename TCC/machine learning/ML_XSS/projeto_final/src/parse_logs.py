import re
import pandas as pd
import joblib
import os
from src.feature_extractor import extrair_features # Certifique-se que este arquivo existe na mesma pasta

def processar_logs_reais(arquivo_entrada, arquivo_saida):
    # 1. Carregar o modelo já treinado com a proporção 70/30
    if not os.path.exists('models/random_forest_v1.pkl'):
        print("Erro: Modelo não encontrado em models/random_forest_v1.pkl. Treine o modelo primeiro.")
        return
    
    modelo = joblib.load('models/random_forest_v1.pkl')
    
    # 2. Regex específica para o padrão do seu Spring Boot
    # Captura: Data, IP, e o conteúdo do BODY
    regex_log = r"(?P<data>\d{4}-\d{2}-\d{2}T[\d:.-]+).*IP:\s*(?P<ip>[\d\.]+).*BODY:\s*(?P<body>\{.*\})\s*\|\s*STATUS"
    
    dados_analisados = []

    print(f"Iniciando análise do arquivo: {arquivo_entrada}")

    if not os.path.exists(arquivo_entrada):
        print(f"Erro: Arquivo {arquivo_entrada} não encontrado.")
        return

    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        for linha in f:
            match = re.search(regex_log, linha)
            if match:
                corpo_requisicao = match.group('body')
                ip_origem = match.group('ip')
                data_hora = match.group('data')
                
                # Passo de ML: Extrair features do body e prever
                features = extrair_features(corpo_requisicao)
                predicao = modelo.predict([features])[0]
                
                dados_analisados.append({
                    'Data/Hora': data_hora,
                    'IP': ip_origem,
                    'Payload': corpo_requisicao,
                    'Classificacao': 'ATAQUE' if predicao == 1 else 'NORMAL'
                })

    # 3. Salvar o relatório final
    df = pd.DataFrame(dados_analisados)
    df.to_csv(arquivo_saida, index=False)
    
    print("-" * 30)
    print("ANÁLISE CONCLUÍDA")
    print(f"Total de logs processados: {len(df)}")
    print(df['Classificacao'].value_counts())
    print(f"Relatório salvo em: {arquivo_saida}")

if __name__ == "__main__":
    # Ajuste os nomes dos arquivos conforme sua pasta data/raw e reports/
    processar_logs_reais('data/app_logs/meus_logs_reais.log', 'reports/resultado_analise.csv')