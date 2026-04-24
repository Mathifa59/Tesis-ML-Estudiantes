# Sistema de Alerta Temprana — Riesgo Académico

Dashboard Streamlit del proyecto **P20261012** (UPC — Taller de Proyectos I, Ingeniería de Sistemas de Información).
Detecta estudiantes en riesgo de bajo rendimiento académico en colegios privados de Lima Metropolitana usando Random Forest + SMOTE.

## Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `dataset_estudiantes.csv` | Dataset de estudiantes (features + target). |
| `modelo_riesgo_academico.ipynb` | Notebook original de entrenamiento y EDA. |
| `train_model.py` | Script que entrena el Random Forest y genera los `.pkl`. |
| `app.py` | Dashboard Streamlit. |
| `requirements.txt` | Dependencias de Python. |

## Cómo ejecutar

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo

```bash
python train_model.py
```

Esto genera tres archivos:

- `modelo_rf.pkl` — modelo Random Forest entrenado.
- `label_encoder.pkl` — encoder de la columna `seccion`.
- `metricas_modelo.pkl` — métricas (accuracy, precision, recall, f1, auc), matriz de confusión, datos ROC e importancia de variables.

### 3. Correr la app

```bash
streamlit run app.py
```

La app abre en `http://localhost:8501`.

## Secciones del dashboard

1. **KPIs globales** — total de estudiantes, número y porcentaje por nivel de riesgo.
2. **Ranking por urgencia** — tabla ordenada de mayor a menor probabilidad, con exportación a CSV.
3. **Visualizaciones** — distribución de riesgo, importancia de variables, matriz de confusión y curva ROC.
4. **Detalle por estudiante** — ficha individual con gauge de probabilidad, top 3 factores y explicación.

La barra lateral permite filtrar por grado, sección y nivel de riesgo en tiempo real.

## Criterio de niveles de riesgo

| Nivel | Probabilidad | Color |
|---|---|---|
| ALTO | ≥ 0.70 | Rojo `#A32D2D` |
| MEDIO | 0.45 – 0.69 | Naranja `#BA7517` |
| BAJO | < 0.45 | Verde `#3B6D11` |

## Hiperparámetros del modelo

Random Forest: `n_estimators=200`, `max_depth=8`, `min_samples_split=5`, `random_state=42`.
Balanceo de clases con SMOTE aplicado únicamente al conjunto de entrenamiento.
