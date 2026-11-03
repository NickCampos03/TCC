import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd

def save_report_as_png(model, X_test, y_test, history=None, vectorizer=None):

    os.makedirs('reports', exist_ok=True)
    
    y_probs = model.predict(X_test, verbose=0)
    y_pred = (y_probs > 0.5).astype("int32").flatten()
    
    # --- 1. RELATÓRIO DE QUALIFICAÇÃO ---
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Ataque'], zero_division=0)
    plt.figure(figsize=(9, 5))
    plt.text(0.01, 0.95, str(report), {'fontsize': 11}, fontproperties='monospace', va='top')
    plt.axis('off')
    plt.title("Relatório de Qualificação - Redes Neurais (XSS Detection)", fontsize=13, weight='bold', pad=20)
    plt.savefig('reports/relatorio_qualificacao_nn.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. MATRIZ DE CONFUSÃO ---
    plt.rcdefaults()
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Ataque'], yticklabels=['Normal', 'Ataque'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.ylabel('Classe Real', fontsize=11, weight='bold')
    plt.xlabel('Classe Prevista', fontsize=11, weight='bold')
    plt.title('Matriz de Confusão - Modelo Neural', fontsize=13, weight='bold', pad=15)
    plt.savefig('reports/matriz_confusao_nn.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 3. CURVAS DE APRENDIZADO ---
    if history is not None and hasattr(history, 'history'):
        plt.style.use('ggplot')
        # Criando duas figuras lado a lado para não espremer os dados
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = np.arange(0, len(history.history['loss']))
        
        # Subplot 1: Loss (Perda)
        ax1.plot(epochs_range, history.history['loss'], label='Treino (Loss)', color='firebrick', linewidth=2)
        ax1.plot(epochs_range, history.history['val_loss'], label='Validação (Loss)', color='dodgerblue', linewidth=2)
        ax1.set_title('Evolução da Perda (Loss)', fontsize=12, weight='bold')
        ax1.set_xlabel('Epoch #')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # Subplot 2: Accuracy (Acurácia)
        acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
        
        ax2.plot(epochs_range, history.history[acc_key], label='Treino (Acc)', color='mediumpurple', linewidth=2)
        ax2.plot(epochs_range, history.history[val_acc_key], label='Validação (Acc)', color='dimgray', linewidth=2)
        ax2.set_title('Evolução da Acurácia (Accuracy)', fontsize=12, weight='bold')
        ax2.set_xlabel('Epoch #')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='lower right')
        ax2.grid(True)
        
        plt.suptitle('Histórico de Treinamento da Rede Neural', fontsize=14, weight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('reports/curva_aprendizado_nn.png', dpi=300, bbox_inches='tight')
        plt.close()

    # # --- 4. GRAFICO DAS TOP 10 FEATURES ---
    if vectorizer is not None:
        plt.rcdefaults()
        
        # Extrai os pesos matemáticos da primeira camada Dense
        pesos_camada_entrada = model.layers[0].get_weights()[0]
        importancia_features = np.sum(np.abs(pesos_camada_entrada), axis=1)
        
        # Recupera os nomes das features limpando espaços em branco nas extremidades
        feature_names = [str(name).strip() for name in vectorizer.get_feature_names_out()]
        
        # Cria o DataFrame inicial
        df_raw = pd.DataFrame({
            'Feature': feature_names,
            'Importancia': importancia_features
        })
        
        # Agrupa por nome de feature e soma os pesos analíticos de forma absoluta
        df_grouped = df_raw.groupby('Feature', as_index=False)['Importancia'].sum()
        
        # Ordena do maior para o menor e pega as 10 principais
        df_grouped = df_grouped.sort_values(by='Importancia', ascending=False).head(10)
        df_grouped = df_grouped.iloc[::-1]
        
        # Geração do gráfico
        plt.figure(figsize=(10, 5))
        plt.barh(df_grouped['Feature'], df_grouped['Importancia'], color='teal', edgecolor='black', height=0.55)
        
        plt.title('Top 10 Tokens Mais Importantes na Tomada de Decisão da IA', fontsize=12, weight='bold', pad=15)
        plt.xlabel('Peso Analítico Acumulado (Soma dos Pesos Neuronais)', fontsize=11)
        plt.ylabel('Tokens Únicos Extraídos (TF-IDF)', fontsize=11)
        plt.xlim(0, df_grouped['Importancia'].max() * 1.05)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig('reports/top_features_nn.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SUCESSO] Gráfico de features corrigido e unificado sem duplicatas!")

    print("--- [SUCESSO] Gráficos otimizados gerados em /reports ---")