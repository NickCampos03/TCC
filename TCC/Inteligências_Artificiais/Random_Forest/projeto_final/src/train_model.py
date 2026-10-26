import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from src.evaluate_model import avaliar_modelo
from src.feature_extractor import FEATURE_COLUMNS
from src.preprocess import PROCESSED_DATASET_PATH


# 1. CONFIGURAÇÕES E CANDIDATOS DE HIPERPARÂMETROS
DATASET_PATH = PROCESSED_DATASET_PATH

RF_CANDIDATES = [
    {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_split": 4,
        "min_samples_leaf": 4,
        "max_features": 0.5,
    },
    {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_split": 20,
        "min_samples_leaf": 2,
        "max_features": 0.5,
    },
    {
        "n_estimators": 300,
        "max_depth": 24,
        "min_samples_split": 12,
        "min_samples_leaf": 8,
        "max_features": 0.5,
    },
]

# 2. FUNÇÕES AUXILIARES (MODELO E DATASET)
def _criar_modelo(params):
    return RandomForestClassifier(**params, bootstrap=True, oob_score=True, class_weight="balanced_subsample", random_state=42, n_jobs=-1)

def _carregar_dataset():
    df = pd.read_csv(DATASET_PATH)
    missing = [col for col in FEATURE_COLUMNS + ["Class"] if col not in df.columns]

    if missing:
        raise ValueError("Dataset processado invalido. Colunas ausentes: " + ", ".join(missing))

    X = df[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(df["Class"], errors="coerce").astype(int)

    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X, y

# 3. SELEÇÃO DE HIPERPARÂMETROS (VALIDAÇÃO CRUZADA)
def _selecionar_hiperparametros(X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    resultados = []

    for idx, params in enumerate(RF_CANDIDATES, start=1):
        print(f"[INFO] Validando candidato {idx}/{len(RF_CANDIDATES)}...")
        model = _criar_modelo(params)
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring={"f1": "f1", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}, return_train_score=True, n_jobs=-1)

        row = {
            **params,
            "mean_train_f1": scores["train_f1"].mean(),
            "mean_test_f1": scores["test_f1"].mean(),
            "mean_test_precision": scores["test_precision"].mean(),
            "mean_test_recall": scores["test_recall"].mean(),
            "mean_test_roc_auc": scores["test_roc_auc"].mean(),
        }
        row["f1_gap_cv"] = row["mean_train_f1"] - row["mean_test_f1"]
        resultados.append(row)

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(by=["mean_test_f1", "f1_gap_cv"], ascending=[False, True])

    os.makedirs("reports", exist_ok=True)
    df_resultados.to_csv("reports/model_selection_rf.csv", index=False)

    best = df_resultados.iloc[0].to_dict()
    best_params = {key: best[key] for key in RF_CANDIDATES[0].keys()}

    if pd.isna(best_params["max_depth"]):
        best_params["max_depth"] = None

    for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
        if best_params[key] is not None:
            best_params[key] = int(best_params[key])

    print("[INFO] Melhores parametros:", best_params)
    print("[INFO] F1 medio CV:", round(float(best["mean_test_f1"]), 5))
    print("[INFO] Gap F1 CV:", round(float(best["f1_gap_cv"]), 5))

    return best_params

# 4. PIPELINE DE EXECUÇÃO DO TREINAMENTO
def executar_treinamento():
    print("\n[INFO] Carregando dataset...")

    X, y = _carregar_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    best_params = _selecionar_hiperparametros(X_train, y_train)
    model = _criar_modelo(best_params)

    print("[INFO] Treinando modelo para avaliacao...")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    gap_f1 = f1_score(y_train, train_pred) - f1_score(y_test, test_pred)

    print("[INFO] Gap F1 treino-teste:", round(float(gap_f1), 5))

    if hasattr(model, "oob_score_"):
        print("[INFO] OOB score:", round(float(model.oob_score_), 5))

    if gap_f1 > 0.05:
        print("[AVISO] Gap de F1 acima de 0.05. Considere aumentar min_samples_leaf ou reduzir max_depth.")

    avaliar_modelo(model, X_train, y_train, X_test, y_test, FEATURE_COLUMNS)
    final_model = _criar_modelo(best_params)
    print("[INFO] Treinando modelo final com todos os dados processados...")
    final_model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/random_forest_v1.pkl")
    joblib.dump(FEATURE_COLUMNS, "models/feature_columns.joblib")

    print("[SUCESSO] Modelo salvo.")

    return final_model