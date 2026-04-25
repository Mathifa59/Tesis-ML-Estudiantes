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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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

    # --- Comparativa de modelos ---
    candidatos = {
        "Random Forest   ": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5,
            random_state=SEED, n_jobs=-1,
        ),
        "Logistic Regr.  ": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree   ": DecisionTreeClassifier(random_state=SEED),
    }

    print()
    print("=" * 65)
    print("  COMPARATIVA DE MODELOS (validación cruzada 5-fold sobre train+SMOTE)")
    print("=" * 65)
    print(f"  {'Modelo':<20} {'CV F1 mean':>10} {'CV F1 std':>10}")
    print("  " + "-" * 42)
    for nombre, clf in candidatos.items():
        cv_scores = cross_val_score(clf, X_train_sm, y_train_sm, cv=5, scoring="f1", n_jobs=-1)
        print(f"  {nombre:<20} {cv_scores.mean():.4f}     ±{cv_scores.std():.4f}")
    print()

    # --- Entrenamiento final con Random Forest ---
    print("Entrenando modelo final: Random Forest...")
    model = candidatos["Random Forest   "]
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

    cv_rf = cross_val_score(model, X_train_sm, y_train_sm, cv=5, scoring="f1", n_jobs=-1)

    print()
    print("=" * 55)
    print("  REPORTE FINAL - RANDOM FOREST (conjunto de prueba)")
    print("=" * 55)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  CV F1 (5-fold): {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
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
        "cv_f1_mean": float(cv_rf.mean()),
        "cv_f1_std": float(cv_rf.std()),
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
