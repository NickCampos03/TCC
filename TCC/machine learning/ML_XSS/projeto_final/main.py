import os
from src.preprocess import processar_e_balancear
from src.train_model import executar_treinamento
from src.parse_logs import processar_logs_reais

def menu():
    print("\n" + "="*30)
    print("SISTEMA DE DETECÇÃO DE XSS (ML)")
    print("="*30)
    print("1. Pré-processar Dados (Limpeza + 70/30)")
    print("2. Treinar Modelo (Random Forest)")
    print("3. Analisar Logs Reais (Relatório)")
    print("4. Executar Pipeline Completa")
    print("0. Sair")
    return input("\nEscolha uma opção: ")

def main():
    # Garante que as pastas necessárias existam
    for pasta in ['data/processed', 'models', 'reports']:
        if not os.path.exists(pasta):
            os.makedirs(pasta)

    while True:
        opcao = menu()

        if opcao == '1':
            print("\n[INFO] Iniciando Pré-processamento...")
            processar_e_balancear()
        
        elif opcao == '2':
            print("\n[INFO] Iniciando Treinamento do Modelo...")
            executar_treinamento()
        
        elif opcao == '3':
            print("\n[INFO] Analisando Logs Reais...")
            # Caminho dos seus logs reais que você nos mandou
            processar_logs_reais('data/app_logs/app_logs.log.txt', 'reports/resultado_analise.csv')
        
        elif opcao == '4':
            print("\n[INFO] Executando Fluxo Completo...")
            processar_e_balancear()
            executar_treinamento()
            processar_logs_reais('data/app_logs/app_logs.log.txt', 'reports/resultado_analise.csv')
            print("\n[SUCESSO] Pipeline finalizada com sucesso!")
        
        elif opcao == '0':
            print("Encerrando...")
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    main()