import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix

def plotar_curva_aprendizado(estimator, X, y):
    """Gera gráfico para analisar Overfitting vs Underfitting"""
    print("[INFO] Gerando Curva de Aprendizado... Isso pode demorar alguns minutos.")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Acurácia de Treino")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Acurácia de Validação (Teste)")

    plt.title("Curva de Aprendizado: Verificação de Overfitting")
    plt.xlabel("Número de Amostras de Treinamento")
    plt.ylabel("Acurácia")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig('reports/curva_aprendizado.png')
    plt.close()
    print("[INFO] Gráfico de curva de aprendizado salvo em /reports/curva_aprendizado.png")

def salvar_graficos(modelo, X_test, y_test, nomes_features, report_texto):
    """Gera e salva as visualizações padrões"""
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # 1. Importância das Features
    plt.figure(figsize=(12, 8))
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)
    plt.title('Importância das Características (Features)')
    plt.barh(range(len(indices)), importancias[indices], color='#2b5a91', align='center')
    plt.yticks(range(len(indices)), [nomes_features[i] for i in indices])
    plt.xlabel('Importância Relativa')
    plt.tight_layout()
    plt.savefig('reports/importancia_features.png')
    plt.close()

    # 2. Matriz de Confusão
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Ataque'], yticklabels=['Normal', 'Ataque'])
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.title('Matriz de Confusão Final')
    plt.savefig('reports/matriz_confusao.png')
    plt.close()

def executar_treinamento():
    input_path = 'data/processed/dataset_distribuido_70_30.csv'
    model_path = 'models/random_forest_v1.pkl'

    if not os.path.exists(input_path):
        print(f"Erro: Arquivo {input_path} não encontrado.")
        return

    df = pd.read_csv(input_path)
    coluna_alvo = 'label' if 'label' in df.columns else 'Class'
    X = df.drop(coluna_alvo, axis=1)
    y = df[coluna_alvo].values
    nomes_colunas = X.columns.tolist()

    # --- AJUSTE ESTRATIFICADO (80/20 e 70/30) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # <--- CRUCIAL: Mantém a proporção 70/30 no treino e no teste
    )

    print(f"Iniciando treinamento com {len(X_train)} amostras e {len(nomes_colunas)} features...")
    print(f"Distribuição de Classes no Treino: {np.bincount(y_train)}")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, model_path)
    
    y_pred = rf.predict(X_test)
    report_texto = classification_report(y_test, y_pred, target_names=['Normal', 'Ataque XSS'])
    print("\n--- Relatório de Classificação ---\n", report_texto)

    # Gera os gráficos padrões
    salvar_graficos(rf, X_test, y_test, nomes_colunas, report_texto)
    
    # Gera a análise de Overfitting usando o dataset INTEIRO (X, y)
    plotar_curva_aprendizado(rf, X, y)

if __name__ == "__main__":
    executar_treinamento()