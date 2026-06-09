import os
import pandas as pd
# Importamos a nova função de unificação
from src.preprocess import processar_dataset, unificar_datasets 
from src.train_model import executar_treinamento
from src.parse_logs import processar_logs_reais

def menu():
    print("\n" + "="*45)
    print("  SISTEMA DE DETECÇÃO DE INTRUSÃO (XSS & SQLI)")
    print("="*45)
    print("1. Pré-processar Datasets (XSS e SQL Injection)")
    print("2. Unificar Dados e Treinar Modelo Mestre")
    print("3. Analisar Logs Reais (Detecção em Tempo Real)")
    print("4. Executar Pipeline Completa (Tudo em 1 Passo)")
    print("0. Sair")
    return input("\nEscolha uma opção: ")

def main():
    # Garante que as pastas de saída existam
    pastas = [
        'data/processed/XSS', 
        'data/processed/SQLI', 
        'models', 
        'reports',
        'data/app_logs'
    ]
    for pasta in pastas:
        os.makedirs(pasta, exist_ok=True)

    while True:
        opcao = menu()

        if opcao == '1':
            print("\n[INFO] Iniciando Pré-processamento de Datasets...")
            processar_dataset('XSS')
            processar_dataset('SQLI')
            # Após processar individualmente, criamos o arquivo mestre
            unificar_datasets()
            print("\n[SUCESSO] Dados processados, balanceados e unificados!")
        
        elif opcao == '2':
            print("\n[INFO] Iniciando Treinamento do Modelo Mestre...")
            # Verifica se o arquivo mestre existe antes de treinar
            if os.path.exists('data/processed/dataset_final_mestre.csv'):
                executar_treinamento()
            else:
                print("[ERRO] Dataset mestre não encontrado! Execute a Opção 1 primeiro.")
        
        elif opcao == '3':
            print("\n[INFO] Analisando Logs Reais com o Modelo Unificado...")
            log_input = 'data/app_logs/app_logs.log.txt'
            relatorio_output = 'reports/resultado_analise.csv'
            
            if os.path.exists(log_input):
                processar_logs_reais(log_input, relatorio_output)
            else:
                print(f"[ERRO] Arquivo de log não encontrado em: {log_input}")
        
        elif opcao == '4':
            print("\n[INFO] Executando Fluxo Completo (XSS + SQLI)...")
            
            # Passo 1: Extração e Limpeza Individual
            processar_dataset('XSS')
            processar_dataset('SQLI')
            
            # Passo 2: Unificação (Crucial para o modelo unificado)
            unificar_datasets()
            
            # Passo 3: Treino do Modelo (Gera o random_forest_v1.pkl)
            executar_treinamento()
            
            # Passo 4: Aplicação em logs reais
            processar_logs_reais('data/app_logs/app_logs.log.txt', 'reports/resultado_analise.csv')
            
            print("\n[SUCESSO] Pipeline Finalizada com Sucesso!")
            print("-> Modelo salvo em: models/random_forest_v1.pkl")
            print("-> Relatório em: reports/resultado_analise.csv")
        
        elif opcao == '0':
            print("Encerrando o sistema...")
            break
        else:
            print("Opção inválida! Tente novamente.")

if __name__ == "__main__":
    main()