import os

import pandas as pd

from scripts.analise_logs import analisar_logs_neural_network
from scripts.evaluate_model import save_report_as_png
from scripts.feature_extractor import FEATURE_COLUMNS
from scripts.preprocess import processar_e_unificar
from scripts.train_neural_net import train_model


def executar_treinamento():
    path = 'data/dataset_xss.csv'
    if not os.path.exists(path):
        print("Erro: Dataset nao encontrado. Execute o passo 1.")
        return

    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]

    if 'label' not in df.columns:
        print("Erro: coluna 'label' nao encontrada no dataset.")
        return

    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        print("Erro: dataset sem as features esperadas:")
        print(", ".join(missing))
        return

    df = df[FEATURE_COLUMNS + ['label']].apply(pd.to_numeric, errors='coerce').dropna()
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]

    if df.empty:
        print("Erro: dataset vazio apos limpeza.")
        return

    X = df[FEATURE_COLUMNS].values
    y = df['label'].values

    model, history, X_test, y_test = train_model(X, y)
    save_report_as_png(model, X_test, y_test, history=history, vectorizer=None)


def main():
    while True:
        print("\n1. Preprocessar | 2. Treinar | 3. Analisar Logs | 0. Sair")
        op = input("Opcao: ")
        if op == '1':
            processar_e_unificar()
        elif op == '2':
            executar_treinamento()
        elif op == '3':
            analisar_logs_neural_network()
        elif op == '0':
            break


if __name__ == "__main__":
    main()
