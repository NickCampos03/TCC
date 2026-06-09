import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

def train_model(payloads_reais, labels_reais):
    print("\n--- [INICIANDO TREINAMENTO DA REDE NEURAL INTELIGENTE] ---")
    print(f"Quantidade de dados recebidos para o treino: {len(payloads_reais)}")
    
    #Força a conversão para formatos puros que o scikit-learn entende perfeitamente
    try:
        if hasattr(payloads_reais, 'tolist'):
            payloads_reais = payloads_reais.tolist()
        else:
            payloads_reais = list(payloads_reais)
            
        if hasattr(labels_reais, 'tolist'):
            labels_reais = labels_reais.tolist()
        else:
            labels_reais = list(labels_reais)
    except Exception as e:
        print(f"[Aviso] Falha na conversão de tipos, tentando seguir adiante... Erro: {e}")

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        payloads_reais, labels_reais, test_size=0.2, random_state=42
    )

    # =========================================================================
    # ENGENHARIA DE RECURSOS AUTOMÁTICA (NLP VIA CHAR N-GRAMS)
    # =========================================================================
    print("Criando Vetorizador TF-IDF baseado em sub-sequências de caracteres...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=1500)
    
    X_train = vectorizer.fit_transform(X_train_raw).toarray()
    X_val = vectorizer.transform(X_val_raw).toarray()
    
    print(f"Dicionário matemático gerado: {X_train.shape[1]} colunas automáticas.")

    # =========================================================================
    # ARQUITETURA DA REDE NEURAL
    # =========================================================================
    print("Construindo topologia da Rede Neural...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.4),  
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # =========================================================================
    # TREINAMENTO COM EARLY STOPPING
    # =========================================================================
    print("Iniciando otimização dos pesos matemáticos...")
    monitoramento = EarlyStopping(
        monitor='val_loss',
        patience=3,             
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, np.array(y_train),
        epochs=40,              
        batch_size=128,         
        validation_data=(X_val, np.array(y_val)),
        callbacks=[monitoramento],
        verbose=1
    )

    if not os.path.exists('models'): os.makedirs('models')
    model.save('models/model_nn.keras')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    print("\n[SUCESSO] Modelo e Vetorizador salvos com dados reais!")
    
    return model, history, X_val, y_val, vectorizer