import pandas as pd
import numpy as np
import os
from scripts.preprocess import processar_e_unificar
from scripts.train_neural_net import train_model
from scripts.analise_logs import analisar_logs_neural_network
from scripts.evaluate_model import save_report_as_png

def executar_treinamento():
    print("\n--- [MODO TREINAMENTO NLP] ---")
    
    path = 'data/dataset_hibrido.csv' 
    
    if not os.path.exists(path):
        print(f"Erro: {path} não encontrado. Certifique-se de unificar os datasets primeiro (Opção 1).")
        return

    df = pd.read_csv(path).dropna()
    
    if 'Class' in df.columns:
        df = df.rename(columns={'Class': 'label'})

    # No modelo NLP, X contém apenas as strings brutas coletadas
    X = df['payload'].values
    y = df['label'].values

    # Treina o modelo usando extração estatística TF-IDF
    model, history, X_test, y_test, vectorizer = train_model(X, y)
    
    # Gera e salva a matriz de confusão e relatório atualizados baseado no novo teste
    print("\nGerando novos relatórios e Matriz de Confusão...")
    
    # --- Passando history e vectorizer explicitamente ---
    save_report_as_png(model, X_test, y_test, history=history, vectorizer=vectorizer)
    
    print("\n--- TREINAMENTO CONCLUÍDO E MODELO SALVO ---")

def menu():
    print("\n" + "="*50)
    print("      IFSP TCC - SISTEMA HÍBRIDO NLP (XSS/SQLI)")
    print("="*50)
    print("[1] Pré-processar Dados (Unificar Texto Bruto XSS + SQLi)")
    print("[2] Treinar Novo Modelo (Rede Neural + TF-IDF)")
    print("[3] Analisar Logs Reais (Detecção com Vetorizador)")
    print("[4] Executar Pipeline Completa (1 -> 2 -> 3)")
    print("[0] Sair")
    return input("\nEscolha uma opção: ")

def main():
    for pasta in ['data', 'models', 'reports']:
        if not os.path.exists(pasta):
            os.makedirs(pasta)

    while True:
        opcao = menu()

        if opcao == '1':
            print("\n[INFO] Iniciando Pré-processamento...")
            processar_e_unificar()
        
        elif opcao == '2':
            print("\n[INFO] Iniciando Treinamento do Modelo...")
            executar_treinamento()
        
        elif opcao == '3':
            print("\n[INFO] Analisando Logs Reais...")
            analisar_logs_neural_network()
        
        elif opcao == '4':
            print("\n[INFO] Executando Fluxo Completo...")
            print("\n--- PASSO 1: Pré-processamento ---")
            processar_e_unificar()
            
            print("\n--- PASSO 2: Treinamento e Avaliação ---")
            executar_treinamento()
            
            print("\n--- PASSO 3: Análise de Logs ---")
            analisar_logs_neural_network()
            
            print("\n[SUCESSO] Pipeline finalizada com sucesso!")
        
        elif opcao == '0':
            print("Encerrando...")
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    main()