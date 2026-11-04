import os

import matplotlib

# Gera gráficos em arquivo sem exigir uma janela Tk/Tcl no ambiente.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


CLASS_NAMES = {
    0: 'Benigno',
    1: 'XSS',
    2: 'SQLi',
    3: 'Path Traversal',
}


def _predizer_classes(model, X_test):
    y_probs = model.predict(X_test, verbose=0)

    if len(y_probs.shape) == 2 and y_probs.shape[1] > 1:
        return np.argmax(y_probs, axis=1), y_probs

    return (y_probs > 0.5).astype("int32").flatten(), y_probs


def _nomes_classes(y_test, y_pred):
    classes = [0, 1, 2, 3]
    return classes, [CLASS_NAMES.get(int(classe), str(classe)) for classe in classes]


def _salvar_erros_matriz(cm, classes, class_names):
    linhas = []
    for i, classe_real in enumerate(classes):
        total_real = cm[i].sum()
        for j, classe_prevista in enumerate(classes):
            if i == j:
                continue

            erros = int(cm[i, j])
            if erros == 0:
                continue

            linhas.append({
                'Classe_Real': class_names[i],
                'Classe_Prevista': class_names[j],
                'Erros': erros,
                'Percentual_na_Classe_Real': round((erros / total_real) * 100, 3) if total_real else 0.0,
            })

    df_erros = pd.DataFrame(linhas)
    if not df_erros.empty:
        df_erros = df_erros.sort_values('Erros', ascending=False)
    df_erros.to_csv('reports/erros_matriz_confusao_nn.csv', index=False)


def save_report_as_png(model, X_test, y_test, history=None, vectorizer=None):
    os.makedirs('reports', exist_ok=True)

    y_test = np.array(y_test, dtype='int32')
    y_pred, _ = _predizer_classes(model, X_test)
    classes, class_names = _nomes_classes(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report_dict).transpose().to_csv('reports/relatorio_qualificacao_nn.csv')

    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    plt.figure(figsize=(10, 5.5))
    plt.text(0.01, 0.95, str(report), {'fontsize': 11}, fontproperties='monospace', va='top')
    plt.axis('off')
    plt.title("Relatorio de Qualificacao - Rede Neural", fontsize=13, weight='bold', pad=20)
    plt.savefig('reports/relatorio_qualificacao_nn.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.rcdefaults()
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv('reports/matriz_confusao_nn.csv')
    _salvar_erros_matriz(cm, classes, class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'size': 16, 'weight': 'bold'},
    )
    plt.ylabel('Classe Real', fontsize=12, weight='bold')
    plt.xlabel('Classe Prevista', fontsize=12, weight='bold')
    plt.title('Matriz de Confusao - Modelo Neural', fontsize=16, weight='bold', pad=18)
    plt.tight_layout()
    plt.savefig('reports/matriz_confusao_nn.png', dpi=300, bbox_inches='tight')
    plt.close()

    if history is not None and hasattr(history, 'history'):
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = np.arange(0, len(history.history['loss']))

        ax1.plot(epochs_range, history.history['loss'], label='Treino (Loss)', color='firebrick', linewidth=2)
        ax1.plot(epochs_range, history.history['val_loss'], label='Validacao (Loss)', color='dodgerblue', linewidth=2)
        ax1.set_title('Evolucao da Perda (Loss)', fontsize=12, weight='bold')
        ax1.set_xlabel('Epoch #')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True)

        acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
        ax2.plot(epochs_range, history.history[acc_key], label='Treino (Acc)', color='mediumpurple', linewidth=2)
        ax2.plot(epochs_range, history.history[val_acc_key], label='Validacao (Acc)', color='dimgray', linewidth=2)
        ax2.set_title('Evolucao da Acuracia (Accuracy)', fontsize=12, weight='bold')
        ax2.set_xlabel('Epoch #')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='lower right')
        ax2.grid(True)

        plt.suptitle('Historico de Treinamento da Rede Neural', fontsize=14, weight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('reports/curva_aprendizado_nn.png', dpi=300, bbox_inches='tight')
        plt.close()

    if vectorizer is not None:
        plt.rcdefaults()

        pesos_camada_entrada = model.layers[0].get_weights()[0]
        importancia_features = np.sum(np.abs(pesos_camada_entrada), axis=1)
        feature_names = [str(name).strip() for name in vectorizer.get_feature_names_out()]

        df_raw = pd.DataFrame({
            'Feature': feature_names,
            'Importancia': importancia_features,
        })
        df_grouped = df_raw.groupby('Feature', as_index=False)['Importancia'].sum()
        df_grouped = df_grouped.sort_values(by='Importancia', ascending=False).head(10).iloc[::-1]

        plt.figure(figsize=(10, 5))
        plt.barh(df_grouped['Feature'], df_grouped['Importancia'], color='teal', edgecolor='black', height=0.55)
        plt.title('Top 10 Tokens Mais Importantes na Tomada de Decisao da IA', fontsize=12, weight='bold', pad=15)
        plt.xlabel('Peso Analitico Acumulado', fontsize=11)
        plt.ylabel('Tokens Extraidos (TF-IDF)', fontsize=11)
        plt.xlim(0, df_grouped['Importancia'].max() * 1.05)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('reports/top_features_nn.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SUCESSO] Grafico de features corrigido e unificado sem duplicatas!")

    print("--- [SUCESSO] Graficos otimizados gerados em /reports ---")
