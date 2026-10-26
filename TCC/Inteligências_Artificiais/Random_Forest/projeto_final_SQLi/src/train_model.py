import pandas as pd
import joblib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    plt.title("Curva de Aprendizado (Modelo Unificado)")
    plt.xlabel("Amostras"), plt.ylabel("Acurácia"), plt.legend()
    plt.savefig('reports/curva_aprendizado.png')
    plt.close('all')

def salvar_graficos(modelo, X_test, y_test):
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(8, 6))
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    # Ajuste para mostrar os nomes das classes no gráfico se forem strings
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.title('Matriz de Confusão (Detecção Multiclasse)')
    plt.xlabel('Predito'), plt.ylabel('Real')
    plt.savefig('reports/matriz_confusao.png')
    plt.close('all')

def executar_treinamento():
    input_path = 'data/processed/dataset_final_mestre.csv'
    
    if not os.path.exists(input_path):
        print(f"[ERRO] Dataset unificado não encontrado.")
        return

    print(f"[INFO] Carregando dataset unificado: {input_path}")
    df = pd.read_csv(input_path)
    
    # --- AJUSTE PARA O TCC: MAPEAR LABELS SE FOREM NUMÉRICOS ---
    # Se o seu preprocess salva 0 e 1, mas você quer nomes:
    if df['label'].dtype != 'object':
        # Exemplo: 0=NORMAL, 1=XSS, 2=SQLI (ajuste conforme seu CSV)
        # Se você só tem 0 e 1, o modelo não vai adivinhar qual ataque é qual.
        # Certifique-se que o preprocess.py passa o nome da classe ou IDs diferentes.
        pass

    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Treinando Random Forest com {len(X_train)} amostras...")
    # Aumentamos a robustez para evitar os falsos positivos que vimos nos seus logs
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=30, 
        min_samples_split=5, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)

    # 4. SALVAR MODELO E METADADOS
    os.makedirs('models', exist_ok=True)
    caminho_modelo = 'models/random_forest_v1.pkl'
    
    # É vital salvar as classes para o script de logs saber o que é '1' ou '0'
    model_data = {
        'model': rf, 
        'features': X.columns.tolist(),
        'target_names': rf.classes_.tolist() 
    }
    joblib.dump(model_data, caminho_modelo)
    
    # 5. RESULTADOS
    print("\n--- Relatório de Classificação ---")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    salvar_graficos(rf, X_test, y_test)
    plotar_curva_aprendizado(rf, X, y)
    
    print(f"\n[SUCESSO] Modelo salvo em: {caminho_modelo}")

if __name__ == "__main__":
    executar_treinamento()