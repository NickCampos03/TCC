import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Mapeamento multiclasse: 0 Benigno, 1 XSS, 2 SQLi, 3 Path Traversal.
CLASS_LABELS = [0, 1, 2, 3]
CLASS_NAMES = ["Benigno", "XSS", "SQLi", "Path Traversal"]


def avaliar_modelo(model, X_train, y_train, X_test, y_test, feature_names):
    os.makedirs("reports", exist_ok=True)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metricas = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision_weighted": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "train_recall_weighted": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "train_f1_weighted": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "test_recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "test_f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "test_roc_auc_ovr_weighted": roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted", labels=CLASS_LABELS),
    }
    metricas["f1_gap_train_test"] = metricas["train_f1_weighted"] - metricas["test_f1_weighted"]

    if hasattr(model, "oob_score_"):
        metricas["oob_score"] = float(model.oob_score_)

    print("\n===== METRICAS =====")
    print("Train F1 :", metricas["train_f1_weighted"])
    print("Test Acc :", metricas["test_accuracy"])
    print("Test Prec:", metricas["test_precision_weighted"])
    print("Test Rec :", metricas["test_recall_weighted"])
    print("Test F1  :", metricas["test_f1_weighted"])
    print("ROC-AUC  :", metricas["test_roc_auc_ovr_weighted"])
    print("F1 Gap   :", metricas["f1_gap_train_test"])

    with open("reports/metricas_rf.json", "w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, indent=2)

    pd.DataFrame([metricas]).to_csv("reports/metricas_rf.csv", index=False)

    report = classification_report(y_test, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES, zero_division=0)
    print("\n===== RELATORIO DE CLASSIFICACAO =====")
    print(report)

    plt.figure(figsize=(10, 6))
    plt.text(0.01, 0.05, report, family="monospace")
    plt.axis("off")
    plt.savefig("reports/relatorio_qualificacao_rf.png", dpi=300, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    print("===== MATRIZ DE CONFUSAO =====")
    print(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")

    ax.set_title("Matriz de Confusao - Modelo Random Forest", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("", fontsize=0)
    ax.set_ylabel("", fontsize=0)
    ax.tick_params(axis="both", which="major", labelsize=14)

    for text in disp.text_.ravel():
        text.set_fontsize(14)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("reports/matriz_confusao_rf.png", dpi=300, bbox_inches="tight")
    plt.close()

    importancias = model.feature_importances_

    df_imp = pd.DataFrame({"Feature": feature_names, "Importancia": importancias})
    df_imp = df_imp.sort_values(by="Importancia", ascending=False).head(10)

    plt.figure(figsize=(12, 8))
    plt.barh(df_imp["Feature"], df_imp["Importancia"])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("reports/top_features_rf.png", dpi=300)
    plt.close()

    print("[SUCESSO] Relatorios gerados.")
