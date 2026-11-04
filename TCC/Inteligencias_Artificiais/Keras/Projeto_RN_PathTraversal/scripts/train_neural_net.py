import os
import random
import re

import joblib
import keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


RANDOM_STATE = 42
MAX_FEATURES = int(os.getenv('TFIDF_MAX_FEATURES', '6000'))
BATCH_SIZE = int(os.getenv('NN_BATCH_SIZE', '512'))
EPOCHS = int(os.getenv('NN_EPOCHS', '25'))
L2_FACTOR = float(os.getenv('NN_L2_FACTOR', '0.002'))
LABEL_SMOOTHING = float(os.getenv('NN_LABEL_SMOOTHING', '0.03'))
N_CLASSES = 4


def _converter_para_listas(payloads_reais, labels_reais):
    try:
        payloads = payloads_reais.tolist() if hasattr(payloads_reais, 'tolist') else list(payloads_reais)
        labels = labels_reais.tolist() if hasattr(labels_reais, 'tolist') else list(labels_reais)
    except Exception as e:
        print(f"[Aviso] Falha na conversao de tipos, tentando seguir adiante... Erro: {e}")
        payloads = list(payloads_reais)
        labels = list(labels_reais)

    payloads = [str(payload) for payload in payloads]
    labels = np.array([int(label) for label in labels], dtype='int32')
    return payloads, labels


def _calcular_class_weight(labels):
    classes = np.unique(labels)
    if len(classes) < 2:
        return None

    pesos = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return {int(classe): float(peso) for classe, peso in zip(classes, pesos)}


def _nomes_classes(labels):
    classes = sorted(np.unique(labels).tolist())
    nomes_padrao = {0: 'Benigno', 1: 'XSS', 2: 'SQLi', 3: 'Path Traversal'}
    return [nomes_padrao.get(int(classe), str(classe)) for classe in classes]


def _aumentar_xss_esparso(payloads, labels):
    """Cria exemplos XSS esparsos para aproximar os descritores de sinais dos logs."""
    rng = random.Random(RANDOM_STATE)
    novos_payloads = []
    novos_labels = []

    for payload, label in zip(payloads, labels):
        if int(label) != 1:
            continue

        # Cada descritor começa por url/html/js e termina em count/bucket N.
        descritores = [
            trecho.strip()
            for trecho in re.split(r'(?=\b(?:url|html|js)\s)', payload)
            if trecho.strip() and re.search(r'\b(?:count|bucket)\s+\d+\s*$', trecho.strip())
        ]
        sinais_xss = [
            item for item in descritores
            if item.startswith((
                'html tag ', 'html event ', 'html attr ', 'js ', 'url tag ',
                'url event ', 'url attr ', 'url cookie ', 'url length ',
                'url special characters ', 'html length ',
            ))
        ]
        if len(sinais_xss) < 2:
            continue

        # Uma amostra por registro, com um subconjunto aleatório dos sinais ativos.
        quantidade = rng.randint(2, min(6, len(sinais_xss)))
        novos_payloads.append(' '.join(rng.sample(sinais_xss, quantidade)))
        novos_labels.append(1)

    if novos_payloads:
        payloads.extend(novos_payloads)
        labels = np.concatenate([labels, np.asarray(novos_labels, dtype='int32')])
        print(f"Aumento de dados XSS: {len(novos_payloads)} exemplos esparsos adicionados ao treino.")

    return payloads, labels


def train_model(payloads_reais, labels_reais):
    print("\n--- [INICIANDO TREINAMENTO DA REDE NEURAL INTELIGENTE] ---")
    print(f"Quantidade de dados recebidos para o treino: {len(payloads_reais)}")

    payloads_reais, labels_reais = _converter_para_listas(payloads_reais, labels_reais)

    if not np.isin(labels_reais, np.arange(N_CLASSES)).all():
        raise ValueError('Os rotulos devem estar no intervalo 0 a 3 (Benigno, XSS, SQLi, Path Traversal).')
    classes_presentes = set(np.unique(labels_reais).tolist())
    classes_esperadas = set(range(N_CLASSES))
    if classes_presentes != classes_esperadas:
        faltantes = sorted(classes_esperadas - classes_presentes)
        raise ValueError(
            f'Treinamento cancelado: faltam exemplos das classes {faltantes}. '
            'Inclua dados de todas as quatro classes antes de treinar.'
        )

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        payloads_reais,
        labels_reais,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=labels_reais,
    )

    X_train_raw, y_train = _aumentar_xss_esparso(list(X_train_raw), y_train)

    print("Criando Vetorizador TF-IDF baseado em subsequencias de caracteres...")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=MAX_FEATURES,
        dtype=np.float32,
        min_df=3,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)

    print(f"Dicionario matematico gerado: {X_train.shape[1]} colunas automaticas.")
    print(f"Matriz TF-IDF de treino: {X_train.shape[0]} linhas esparsas.")

    saida_multiclasse = True

    print("Construindo topologia da Rede Neural...")
    model = Sequential([
        Input(shape=(X_train.shape[1],), sparse=True),
        Dense(96, activation='relu', kernel_regularizer=l2(L2_FACTOR)),
        Dropout(0.55),
        Dense(48, activation='relu', kernel_regularizer=l2(L2_FACTOR)),
        Dropout(0.4),
        # ALTERAÇÃO DA CAMADA DE SAÍDA: quatro classes, conforme os rótulos 0 a 3.
        Dense(4, activation='softmax'),
    ])

    # ALTERAÇÃO DA ARQUITETURA: MLP multiclasse com rótulos inteiros e loss sparse.
    loss = keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0007),
        loss=loss,
        metrics=['accuracy'],
    )

    print("Iniciando otimizacao dos pesos matematicos...")
    monitoramento = EarlyStopping(
        monitor='val_loss',
        patience=2,
        min_delta=0.0005,
        restore_best_weights=True,
        verbose=1,
    )
    reduzir_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=0.00005,
        verbose=1,
    )
    class_weight = _calcular_class_weight(y_train)
    if class_weight:
        print(f"Pesos de classe aplicados: {class_weight}")

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[reduzir_lr, monitoramento],
        class_weight=class_weight,
        verbose=2,
    )

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    model.save('models/model_nn.keras')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(_nomes_classes(labels_reais), 'models/class_names.joblib')
    pd.DataFrame(history.history).to_csv('reports/training_history_nn.csv', index=False)
    print("\n[SUCESSO] Modelo e Vetorizador salvos com dados reais!")

    return model, history, X_val, y_val, vectorizer
