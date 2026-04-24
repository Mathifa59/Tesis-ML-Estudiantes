"""
Entrenamiento del modelo Random Forest para predicción de riesgo académico.
Proyecto P20261012 - UPC Taller de Proyectos I.

Uso:
    python train_model.py

Genera:
    modelo_rf.pkl        -> Random Forest entrenado
    label_encoder.pkl    -> LabelEncoder de la columna 'seccion'
    metricas_modelo.pkl  -> Diccionario con métricas del modelo en test
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


SEED = 42
DATASET_PATH = "dataset_estudiantes.csv"
MODEL_PATH = "modelo_rf.pkl"
ENCODER_PATH = "label_encoder.pkl"
METRICS_PATH = "metricas_modelo.pkl"

FEATURES = [
    "promedio_notas",
    "nota_matematica",
    "nota_comunicacion",
    "porcentaje_asistencia",
    "nivel_conducta",
    "nivel_participacion",
    "tendencia_notas",
    "grado",
    "seccion",
]
TARGET = "riesgo_academico"


def main() -> None:
    np.random.seed(SEED)

    print(f"Cargando dataset desde {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {df.shape[0]} registros, {df.shape[1]} columnas")

    df_model = df.copy()
    le_seccion = LabelEncoder()
    df_model["seccion"] = le_seccion.fit_transform(df_model["seccion"])

    X = df_model[FEATURES]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print("Aplicando SMOTE al conjunto de entrenamiento...")
    smote = SMOTE(random_state=SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"  Train original: {X_train.shape[0]} | Train SMOTE: {X_train_sm.shape[0]}")

    print("Entrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train_sm, y_train_sm)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print()
    print("=" * 55)
    print("  REPORTE FINAL - RANDOM FOREST")
    print("=" * 55)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print()

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "confusion_matrix": cm.tolist(),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "features": FEATURES,
        "feature_importances": model.feature_importances_.tolist(),
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le_seccion, ENCODER_PATH)
    joblib.dump(metrics, METRICS_PATH)

    print(f"Modelo guardado      -> {MODEL_PATH}")
    print(f"LabelEncoder guardado -> {ENCODER_PATH}")
    print(f"Métricas guardadas    -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
