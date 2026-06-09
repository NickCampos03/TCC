import os
from src.parse_logs import processar_logs_reais
from src.preprocess import processar_e_balancear
from src.train_model import executar_treinamento


LOGS_PATH = "data/app_logs/app_logs.log.txt"
RESULTADO_PATH = "reports/resultado_analise.csv"

def menu():
    print("\n" + "=" * 50)
    print("DETECCAO DE XSS COM RANDOM FOREST")
    print("=" * 50)
    print("1 - Pre-processar dataset")
    print("2 - Treinar modelo")
    print("3 - Analisar logs")
    print("4 - Pipeline completa")
    print("0 - Sair")
    return input("\nEscolha: ")


def executar_pipeline_completa():
    processar_e_balancear()
    executar_treinamento()
    processar_logs_reais(
        LOGS_PATH,
        RESULTADO_PATH
    )
    print("\n[SUCESSO] Pipeline concluida.")


def main():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    while True:
        opcao = menu()

        if opcao == "1":
            processar_e_balancear()
        elif opcao == "2":
            executar_treinamento()
        elif opcao == "3":
            processar_logs_reais(
                LOGS_PATH,
                RESULTADO_PATH
            )
        elif opcao == "4":
            executar_pipeline_completa()
        elif opcao == "0":
            break
        else:
            print("Opcao invalida.")


if __name__ == "__main__":
    main()
