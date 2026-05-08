import pandas as pd
import joblib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configuração obrigatória ANTES do pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix

def plotar_curva_aprendizado(estimator, X, y):
    print("[INFO] Gerando Curva de Aprendizado...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Treino")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validação")
    plt.title("Curva de Aprendizado")
    plt.xlabel("Amostras"), plt.ylabel("Acurácia"), plt.legend()
    plt.savefig('reports/curva_aprendizado.png')
    plt.close('all')

def salvar_graficos(modelo, X_test, y_test, nomes_features):
    os.makedirs('reports', exist_ok=True)
    # Matriz de Confusão
    plt.figure(figsize=(8, 6))
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.savefig('reports/matriz_confusao.png')
    plt.close('all')

def executar_treinamento():
    input_path = 'data/processed/dataset_distribuido_70_30.csv'
    df = pd.read_csv(input_path)
    coluna_alvo = 'label' if 'label' in df.columns else 'Class'
    
    X = df.drop(coluna_alvo, axis=1)
    y = df[coluna_alvo]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/random_forest_v1.pkl')
    
    print("\n--- Relatório ---")
    print(classification_report(y_test, rf.predict(X_test)))

    salvar_graficos(rf, X_test, y_test, X.columns)
    plotar_curva_aprendizado(rf, X, y)

if __name__ == "__main__":
    executar_treinamento()