import os
import joblib
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scripts.feature_extractor import FEATURE_COLUMNS


def train_model(X, y):
    print("\n--- [TREINAMENTO OTIMIZADO] ---")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32).reshape(-1)

    if X.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Quantidade de features invalida: {X.shape[1]}. Esperado: {len(FEATURE_COLUMNS)}.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("A coluna label deve conter apenas 0 (normal) e 1 (ataque).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler_xss.joblib')
    joblib.dump(FEATURE_COLUMNS, 'models/feature_columns.joblib')

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(32, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes.tolist(), weights.tolist()))
    epochs = 25
    batch_size = 256
    monitor = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        validation_data=(X_test, y_test),
        callbacks=[monitor],
        verbose=1,
    )

    model.save('models/model_nn.keras')
    final_val_accuracy = history.history.get('val_accuracy', [None])[-1]
    if final_val_accuracy is not None:
        print(f"Treinamento concluido. Val accuracy: {final_val_accuracy:.4f}")
    return model, history, X_test, y_test
