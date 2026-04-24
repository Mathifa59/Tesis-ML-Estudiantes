"""
Sistema de Alerta Temprana - Riesgo Académico
Dashboard Streamlit del proyecto P20261012 (UPC).
"""

from __future__ import annotations

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ── Constants ────────────────────────────────────────────────────────────────

DATASET_PATH = "dataset_estudiantes.csv"
MODEL_PATH   = "modelo_rf.pkl"
ENCODER_PATH = "label_encoder.pkl"
METRICS_PATH = "metricas_modelo.pkl"

FEATURES = [
    "promedio_notas", "nota_matematica", "nota_comunicacion",
    "porcentaje_asistencia", "nivel_conducta", "nivel_participacion",
    "tendencia_notas", "grado", "seccion",
]

COLOR_ALTO  = "#A32D2D"
COLOR_MEDIO = "#BA7517"
COLOR_BAJO  = "#3B6D11"
COLOR_MAP   = {"ALTO": COLOR_ALTO, "MEDIO": COLOR_MEDIO, "BAJO": COLOR_BAJO}

NIVEL_BADGE = {"ALTO": "🔴 ALTO", "MEDIO": "🟡 MEDIO", "BAJO": "🟢 BAJO"}

FEATURE_LABELS = {
    "promedio_notas":        "Promedio de notas",
    "nota_matematica":       "Nota de matemática",
    "nota_comunicacion":     "Nota de comunicación",
    "porcentaje_asistencia": "Porcentaje de asistencia",
    "nivel_conducta":        "Nivel de conducta",
    "nivel_participacion":   "Nivel de participación",
    "tendencia_notas":       "Tendencia de notas",
    "grado":                 "Grado",
    "seccion":               "Sección",
}

FEATURE_ICONS = {
    "promedio_notas":        "📊",
    "nota_matematica":       "🔢",
    "nota_comunicacion":     "📝",
    "porcentaje_asistencia": "📅",
    "nivel_conducta":        "🤝",
    "nivel_participacion":   "🙋",
    "tendencia_notas":       "📈",
    "grado":                 "🏫",
    "seccion":               "🏷️",
}

PLOTLY_BASE = dict(
    template="plotly_white",
    font_family="Inter, sans-serif",
    font_color="#1e293b",
    title_font_color="#0f172a",
    plot_bgcolor="white",
    paper_bgcolor="white",
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Alerta Temprana - Riesgo Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3        { color: #f1f5f9 !important; }
[data-testid="stSidebar"] hr        { border-color: rgba(255,255,255,.12) !important; }

[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #185FA5, #1e6fc2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: .3px;
    transition: all .2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 4px 14px rgba(24,95,165,.5) !important;
    transform: translateY(-1px);
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 500; font-size: 14px; color: #64748b;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #185FA5 !important; font-weight: 600;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #334155 !important;
    font-weight: 500 !important;
    transition: all .2s;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
}

hr { border-color: #e2e8f0; margin: 1.4rem 0; }
h2 { color: #0f172a; font-weight: 700 !important; }
h3 { color: #1e293b; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


# ── Core logic ────────────────────────────────────────────────────────────────

def clasificar_nivel(prob: float) -> str:
    if prob >= 0.70: return "ALTO"
    if prob >= 0.45: return "MEDIO"
    return "BAJO"


@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    metrics = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else {}
    return model, encoder, metrics


@st.cache_data(show_spinner="Cargando dataset...")
def cargar_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df.insert(0, "id_estudiante", [f"EST-{i:04d}" for i in range(1, len(df) + 1)])
    return df


@st.cache_data(show_spinner="Calculando predicciones...")
def generar_predicciones(df: pd.DataFrame, _model, _encoder) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["seccion"] = _encoder.transform(X["seccion"])
    probs = _model.predict_proba(X)[:, 1]
    out   = df.copy()
    out["probabilidad_riesgo"] = probs
    out["prediccion_riesgo"]   = _model.predict(X)
    out["nivel_riesgo"]        = [clasificar_nivel(p) for p in probs]
    return out


def aplicar_filtros(df: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    filt = df.copy()
    if filtros["grados"]:    filt = filt[filt["grado"].isin(filtros["grados"])]
    if filtros["secciones"]: filt = filt[filt["seccion"].isin(filtros["secciones"])]
    if filtros["niveles"]:   filt = filt[filt["nivel_riesgo"].isin(filtros["niveles"])]
    return filt


def verificar_archivos() -> bool:
    faltan = [p for p in (MODEL_PATH, ENCODER_PATH) if not os.path.exists(p)]
    if faltan:
        st.error(
            "No se encontraron los archivos del modelo: "
            + ", ".join(faltan)
            + ". Ejecuta primero **`python train_model.py`**."
        )
        return False
    if not os.path.exists(DATASET_PATH):
        st.error(f"No se encontró `{DATASET_PATH}` en la carpeta del proyecto.")
        return False
    return True


def fmt_trained_at(metrics: dict) -> str:
    raw = metrics.get("trained_at", "—")
    if raw == "—":
        return raw
    try:
        return datetime.fromisoformat(raw).strftime("%d/%m/%Y %H:%M")
    except (ValueError, TypeError):
        return raw


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(df_pred: pd.DataFrame, metrics: dict) -> dict:
    st.sidebar.markdown("""
    <div style="padding:8px 0 14px">
        <div style="color:#93c5fd;font-size:10px;font-weight:700;letter-spacing:2.5px;
             text-transform:uppercase;margin-bottom:6px">Sistema de Predicción</div>
        <div style="color:#f1f5f9;font-size:20px;font-weight:700">Alerta Temprana</div>
        <div style="color:#475569;font-size:12px;margin-top:3px">Riesgo Académico · UPC</div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    st.sidebar.markdown(
        '<p style="color:#64748b;font-size:10px;font-weight:700;'
        'letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px">Filtros</p>',
        unsafe_allow_html=True,
    )

    grados_disp   = sorted(df_pred["grado"].unique().tolist())
    secciones_disp = sorted(df_pred["seccion"].unique().tolist())

    grados_sel    = st.sidebar.multiselect("Grado",         grados_disp,    default=grados_disp)
    secciones_sel = st.sidebar.multiselect("Sección",       secciones_disp, default=secciones_disp)
    niveles_sel   = st.sidebar.multiselect("Nivel de riesgo", ["ALTO", "MEDIO", "BAJO"],
                                           default=["ALTO", "MEDIO", "BAJO"])
    st.sidebar.divider()

    if st.sidebar.button("🔄 Actualizar predicciones", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Mini resumen
    total   = len(df_pred)
    n_alto  = int((df_pred["nivel_riesgo"] == "ALTO").sum())
    pct_alto = n_alto / total * 100 if total else 0

    st.sidebar.markdown(f"""
    <div style="margin-top:16px;background:rgba(255,255,255,.05);border-radius:10px;padding:14px 16px">
        <div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:1px;
             text-transform:uppercase;margin-bottom:10px">Resumen global</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="color:#cbd5e1;font-size:12px">Total estudiantes</span>
            <span style="color:#f1f5f9;font-weight:600;font-size:12px">{total}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:10px">
            <span style="color:#cbd5e1;font-size:12px">En riesgo ALTO</span>
            <span style="color:#fca5a5;font-weight:600;font-size:12px">{n_alto} ({pct_alto:.1f}%)</span>
        </div>
        <div style="height:5px;background:rgba(255,255,255,.1);border-radius:3px">
            <div style="height:5px;width:{min(pct_alto, 100):.1f}%;background:{COLOR_ALTO};border-radius:3px"></div>
        </div>
    </div>
    <div style="margin-top:14px;color:#334155;font-size:11px;text-align:center">
        Modelo actualizado: {fmt_trained_at(metrics)}
    </div>
    """, unsafe_allow_html=True)

    return {"grados": grados_sel, "secciones": secciones_sel, "niveles": niveles_sel}


# ── Header ────────────────────────────────────────────────────────────────────

def render_header(metrics: dict) -> None:
    acc = metrics.get("accuracy", 0)
    auc = metrics.get("auc_roc", 0)
    f1  = metrics.get("f1_score", 0)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 55%,#185FA5 100%);
         padding:28px 36px;border-radius:16px;margin-bottom:26px;
         box-shadow:0 4px 24px rgba(15,23,42,.25)">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;
             flex-wrap:wrap;gap:16px">
            <div>
                <div style="color:#93c5fd;font-size:10px;font-weight:700;letter-spacing:2.5px;
                     text-transform:uppercase;margin-bottom:8px">🎓 Sistema de Alerta Temprana</div>
                <div style="color:white;font-size:28px;font-weight:700;line-height:1.2">
                    Predicción de Riesgo Académico
                </div>
                <div style="color:#94a3b8;font-size:13px;margin-top:8px">
                    Colegios Privados de Lima Metropolitana &nbsp;·&nbsp; Proyecto P20261012 — UPC
                </div>
            </div>
            <div style="display:flex;gap:14px;align-items:stretch;flex-wrap:wrap">
                {"".join([
                    f'<div style="background:rgba(255,255,255,.09);border-radius:12px;'
                    f'padding:14px 20px;text-align:center;min-width:80px">'
                    f'<div style="color:#93c5fd;font-size:10px;font-weight:700;letter-spacing:1px;'
                    f'text-transform:uppercase;margin-bottom:4px">{name}</div>'
                    f'<div style="color:white;font-size:24px;font-weight:700">{val:.3f}</div></div>'
                    for name, val in [("Accuracy", acc), ("F1-Score", f1), ("AUC-ROC", auc)]
                ])}
                <div style="background:rgba(255,255,255,.09);border-radius:12px;
                     padding:14px 20px;text-align:center;min-width:80px">
                    <div style="color:#93c5fd;font-size:10px;font-weight:700;letter-spacing:1px;
                         text-transform:uppercase;margin-bottom:4px">Modelo</div>
                    <div style="color:white;font-size:13px;font-weight:600">Random Forest</div>
                    <div style="color:#64748b;font-size:11px;margin-top:2px">+ SMOTE</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── KPIs ──────────────────────────────────────────────────────────────────────

def tarjeta_kpi(titulo: str, numero: int, total: int, color: str, icon: str) -> None:
    pct = f"{numero/total*100:.1f}%" if total else "0%"
    st.markdown(f"""
    <div style="background:white;border-radius:14px;padding:22px 24px;
         border-top:4px solid {color};
         box-shadow:0 1px 4px rgba(0,0,0,.06),0 6px 20px rgba(0,0,0,.04)">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
            <span style="font-size:18px">{icon}</span>
            <span style="font-size:10px;font-weight:700;color:#94a3b8;
                 text-transform:uppercase;letter-spacing:1px">{titulo}</span>
        </div>
        <div style="font-size:44px;font-weight:700;color:#0f172a;line-height:1">{numero}</div>
        <div style="font-size:13px;color:{color};font-weight:600;margin-top:8px">{pct} del total</div>
    </div>
    """, unsafe_allow_html=True)


def seccion_kpis(df_filt: pd.DataFrame) -> None:
    total  = len(df_filt)
    n_alto = int((df_filt["nivel_riesgo"] == "ALTO").sum())
    n_med  = int((df_filt["nivel_riesgo"] == "MEDIO").sum())
    n_bajo = int((df_filt["nivel_riesgo"] == "BAJO").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: tarjeta_kpi("Total analizados", total,  total, "#185FA5", "👥")
    with c2: tarjeta_kpi("Riesgo alto",      n_alto, total, COLOR_ALTO,  "🔴")
    with c3: tarjeta_kpi("Riesgo medio",     n_med,  total, COLOR_MEDIO, "🟡")
    with c4: tarjeta_kpi("Riesgo bajo",      n_bajo, total, COLOR_BAJO,  "🟢")


# ── Ranking ───────────────────────────────────────────────────────────────────

def seccion_ranking(df_filt: pd.DataFrame) -> None:
    col_h, col_badge = st.columns([3, 1])
    with col_h:
        st.subheader("📋 Ranking de estudiantes por urgencia")
        st.caption("Ordenado de mayor a menor probabilidad de riesgo. Prioriza la atención sobre los primeros.")
    with col_badge:
        if not df_filt.empty:
            n = int((df_filt["nivel_riesgo"] == "ALTO").sum())
            st.markdown(f"""
            <div style="background:#FEF2F2;border:1.5px solid #FECACA;border-radius:10px;
                 padding:12px 16px;text-align:center;margin-top:6px">
                <div style="color:#991B1B;font-size:10px;font-weight:700;letter-spacing:.5px">
                    ATENCIÓN INMEDIATA
                </div>
                <div style="color:{COLOR_ALTO};font-size:30px;font-weight:700;line-height:1.1">{n}</div>
                <div style="color:#991B1B;font-size:11px">estudiantes</div>
            </div>
            """, unsafe_allow_html=True)

    if df_filt.empty:
        st.info("No hay estudiantes que coincidan con los filtros seleccionados.")
        return

    ranking = df_filt.sort_values("probabilidad_riesgo", ascending=False).copy()
    ranking["Tendencia"]       = ranking["tendencia_notas"].map({-1: "↓ Empeora", 0: "→ Estable", 1: "↑ Mejora"})
    ranking["Prob. riesgo"]    = (ranking["probabilidad_riesgo"] * 100).round(1)
    ranking["Nivel de riesgo"] = ranking["nivel_riesgo"].map(NIVEL_BADGE)

    tabla = ranking[[
        "id_estudiante", "grado", "seccion",
        "promedio_notas", "porcentaje_asistencia",
        "nivel_conducta", "nivel_participacion",
        "Tendencia", "Prob. riesgo", "Nivel de riesgo",
    ]].rename(columns={
        "id_estudiante":         "ID",
        "grado":                 "Grado",
        "seccion":               "Sección",
        "promedio_notas":        "Promedio",
        "porcentaje_asistencia": "Asistencia %",
        "nivel_conducta":        "Conducta",
        "nivel_participacion":   "Participación",
    })

    st.dataframe(
        tabla,
        use_container_width=True,
        height=430,
        hide_index=True,
        column_config={
            "Prob. riesgo": st.column_config.ProgressColumn(
                "Prob. riesgo",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "Promedio":      st.column_config.NumberColumn("Promedio",     format="%.2f"),
            "Asistencia %":  st.column_config.NumberColumn("Asistencia %", format="%.1f%%"),
            "Nivel de riesgo": st.column_config.TextColumn("Nivel"),
        },
    )

    csv_bytes = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Exportar ranking a CSV",
        data=csv_bytes,
        file_name="ranking_riesgo_academico.csv",
        mime="text/csv",
    )


# ── Visualizaciones ───────────────────────────────────────────────────────────

def seccion_visualizaciones(df_filt: pd.DataFrame, metrics: dict) -> None:
    st.subheader("📊 Visualizaciones del modelo")

    tab1, tab2, tab3, tab4 = st.tabs([
        "  Distribución de riesgo  ",
        "  Importancia de variables  ",
        "  Matriz de confusión  ",
        "  Curva ROC  ",
    ])

    with tab1:
        if df_filt.empty:
            st.info("Sin datos para graficar con los filtros actuales.")
        else:
            conteo = (
                df_filt["nivel_riesgo"]
                .value_counts()
                .reindex(["ALTO", "MEDIO", "BAJO"])
                .fillna(0)
                .reset_index()
            )
            conteo.columns = ["Nivel", "Cantidad"]
            col_a, col_b = st.columns(2)
            with col_a:
                fig_pie = px.pie(
                    conteo, names="Nivel", values="Cantidad",
                    color="Nivel", color_discrete_map=COLOR_MAP, hole=0.5,
                )
                fig_pie.update_traces(
                    textposition="outside", textinfo="label+percent",
                    pull=[0.04, 0.02, 0],
                )
                fig_pie.update_layout(
                    **PLOTLY_BASE,
                    title=dict(text="Proporción por nivel de riesgo", font_size=15),
                    showlegend=False, height=360,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_b:
                fig_bar = px.bar(
                    conteo, x="Nivel", y="Cantidad",
                    color="Nivel", color_discrete_map=COLOR_MAP, text="Cantidad",
                )
                fig_bar.update_traces(textposition="outside", marker_line_width=0, width=0.45)
                fig_bar.update_layout(
                    **PLOTLY_BASE,
                    title=dict(text="Estudiantes por nivel de riesgo", font_size=15),
                    showlegend=False, height=360,
                    yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        importancias = metrics.get("feature_importances")
        if not importancias:
            st.info("No hay métricas de importancia disponibles.")
        else:
            df_imp = pd.DataFrame({
                "Variable":    [FEATURE_LABELS.get(f, f) for f in FEATURES],
                "Importancia": importancias,
            }).sort_values("Importancia", ascending=True)

            q33 = df_imp["Importancia"].quantile(0.33)
            q67 = df_imp["Importancia"].quantile(0.67)
            colors = [
                COLOR_ALTO if v >= q67 else COLOR_MEDIO if v >= q33 else "#94a3b8"
                for v in df_imp["Importancia"]
            ]

            fig_imp = go.Figure(go.Bar(
                x=df_imp["Importancia"],
                y=df_imp["Variable"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in df_imp["Importancia"]],
                textposition="outside",
            ))
            fig_imp.update_layout(
                **PLOTLY_BASE,
                title=dict(text="Importancia de variables — Random Forest", font_size=15),
                height=480,
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
                yaxis=dict(showgrid=False),
                margin=dict(l=180, r=60, t=52, b=16),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("🔴 Alta influencia · 🟡 Influencia media · ⚫ Baja influencia")

    with tab3:
        cm = metrics.get("confusion_matrix")
        if not cm:
            st.info("No hay matriz de confusión disponible.")
        else:
            cm_arr = np.array(cm)
            labels = ["Sin riesgo", "En riesgo"]
            cm_labels = {(0,0): "VN", (0,1): "FP", (1,0): "FN ⚠", (1,1): "VP ✓"}

            annotations = [
                dict(
                    text=f"<b>{cm_arr[i,j]}</b><br>{cm_labels[(i,j)]}",
                    x=labels[j], y=labels[i],
                    showarrow=False,
                    font=dict(
                        size=18,
                        color="white" if cm_arr[i,j] >= cm_arr.max() * 0.5 else "#1e293b"
                    ),
                )
                for i in range(2) for j in range(2)
            ]

            fig_cm = go.Figure(go.Heatmap(
                z=cm_arr, x=labels, y=labels,
                colorscale=[[0, "#dbeafe"], [0.5, "#3b82f6"], [1, "#1e3a5f"]],
                showscale=False,
            ))
            fig_cm.update_layout(
                **PLOTLY_BASE,
                title=dict(text="Matriz de confusión (conjunto de prueba)", font_size=15),
                xaxis_title="Predicción del modelo",
                yaxis_title="Valor real",
                height=380,
                annotations=annotations,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            cm_info = [
                ("VN",    cm_arr[0,0], "Correctamente sin riesgo", "#3B6D11"),
                ("FP",    cm_arr[0,1], "Alarma innecesaria",        "#BA7517"),
                ("FN ⚠", cm_arr[1,0], "Riesgo NO detectado",       "#A32D2D"),
                ("VP ✓", cm_arr[1,1], "Riesgo detectado",          "#185FA5"),
            ]
            for col, (lbl, val, desc, clr) in zip([c1, c2, c3, c4], cm_info):
                with col:
                    st.markdown(f"""
                    <div style="background:white;border-radius:10px;padding:14px 16px;
                         border-left:4px solid {clr};
                         box-shadow:0 1px 4px rgba(0,0,0,.06)">
                        <div style="font-size:11px;color:#64748b;font-weight:600">{lbl}</div>
                        <div style="font-size:28px;font-weight:700;color:{clr}">{int(val)}</div>
                        <div style="font-size:11px;color:#94a3b8">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab4:
        fpr = metrics.get("roc_fpr")
        tpr = metrics.get("roc_tpr")
        auc = metrics.get("auc_roc")
        if not fpr or not tpr:
            st.info("No hay datos de curva ROC disponibles.")
        else:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"Random Forest (AUC = {auc:.3f})",
                line=dict(color="#185FA5", width=3),
                fill="tozeroy", fillcolor="rgba(24,95,165,0.08)",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Clasificador aleatorio",
                line=dict(color="#94a3b8", dash="dot", width=1.5),
            ))
            fig_roc.add_annotation(
                x=0.72, y=0.22,
                text=f"AUC = <b>{auc:.3f}</b>",
                showarrow=False,
                bgcolor="rgba(24,95,165,0.9)",
                font=dict(color="white", size=14),
                borderpad=10,
            )
            fig_roc.update_layout(
                **PLOTLY_BASE,
                title=dict(text=f"Curva ROC — AUC = {auc:.3f}", font_size=15),
                xaxis=dict(title="Tasa de falsos positivos",
                           showgrid=True, gridcolor="#f1f5f9", range=[0, 1]),
                yaxis=dict(title="Tasa de verdaderos positivos",
                           showgrid=True, gridcolor="#f1f5f9", range=[0, 1.02]),
                height=450,
                legend=dict(x=0.55, y=0.08),
            )
            st.plotly_chart(fig_roc, use_container_width=True)


# ── Detalle por estudiante ────────────────────────────────────────────────────

def _valor_texto(feat: str, val, tendencia_txt: str) -> str:
    if feat == "tendencia_notas":       return tendencia_txt
    if feat == "porcentaje_asistencia": return f"{val:.1f}%"
    if feat == "seccion":               return str(val)
    if feat in ("nivel_conducta", "nivel_participacion"): return f"{int(val)} / 5"
    if feat == "grado":                 return f"{int(val)}°"
    return f"{val:.2f}"


def construir_explicacion(
    est: pd.Series, top_features: list[str], nivel: str, prob: float, tendencia_txt: str
) -> str:
    razones: list[str] = []
    for feat in top_features:
        val = est[feat]
        if feat == "promedio_notas":
            adj = "bajo " if val < 13 else ""
            razones.append(f"un <strong>promedio {adj}({val:.2f}/20)</strong>")
        elif feat == "porcentaje_asistencia":
            adj = "baja " if val < 75 else ""
            razones.append(f"una <strong>asistencia {adj}({val:.1f}%)</strong>")
        elif feat == "nota_matematica":
            razones.append(f"nota de matemática de <strong>{val:.2f}/20</strong>")
        elif feat == "nota_comunicacion":
            razones.append(f"nota de comunicación de <strong>{val:.2f}/20</strong>")
        elif feat == "nivel_conducta":
            razones.append(f"conducta <strong>{int(val)}/5</strong>")
        elif feat == "nivel_participacion":
            razones.append(f"participación <strong>{int(val)}/5</strong>")
        elif feat == "tendencia_notas":
            razones.append(f"tendencia <strong>{tendencia_txt.lower()}</strong>")
        elif feat == "grado":
            razones.append(f"grado <strong>{int(val)}</strong>")
        elif feat == "seccion":
            razones.append(f"sección <strong>{val}</strong>")

    if not razones:
        motivos = "múltiples factores"
    elif len(razones) == 1:
        motivos = razones[0]
    else:
        motivos = ", ".join(razones[:-1]) + f" y {razones[-1]}"

    return (
        f"Este estudiante presenta riesgo <strong>{nivel}</strong> "
        f"({prob*100:.1f}% de probabilidad), principalmente debido a {motivos}."
    )


def seccion_detalle_estudiante(df_filt: pd.DataFrame, metrics: dict) -> None:
    st.subheader("🔍 Detalle por estudiante")

    if df_filt.empty:
        st.info("No hay estudiantes en el filtro actual para inspeccionar.")
        return

    opciones = df_filt.sort_values("probabilidad_riesgo", ascending=False)["id_estudiante"].tolist()
    seleccion = st.selectbox("Selecciona un estudiante (ordenados por mayor riesgo)", opciones)

    est         = df_filt[df_filt["id_estudiante"] == seleccion].iloc[0]
    prob        = float(est["probabilidad_riesgo"])
    nivel       = est["nivel_riesgo"]
    color_nivel = COLOR_MAP[nivel]
    tendencia_txt = {-1: "Empeorando ↓", 0: "Estable →", 1: "Mejorando ↑"}[int(est["tendencia_notas"])]

    # Tarjeta de cabecera
    st.markdown(f"""
    <div style="background:white;border-radius:14px;padding:20px 26px;
         border-left:6px solid {color_nivel};
         box-shadow:0 1px 4px rgba(0,0,0,.06),0 6px 20px rgba(0,0,0,.04);
         margin-bottom:20px;display:flex;align-items:center;gap:18px">
        <div style="background:{color_nivel}22;width:52px;height:52px;border-radius:50%;
             display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0">
            {"🔴" if nivel == "ALTO" else "🟡" if nivel == "MEDIO" else "🟢"}
        </div>
        <div>
            <div style="font-size:20px;font-weight:700;color:#0f172a">{seleccion}</div>
            <div style="display:flex;align-items:center;gap:10px;margin-top:5px;flex-wrap:wrap">
                <span style="background:{color_nivel};color:white;padding:3px 12px;
                     border-radius:20px;font-size:12px;font-weight:700">Riesgo {nivel}</span>
                <span style="color:#64748b;font-size:13px">
                    Grado {int(est['grado'])} — Sección {est['seccion']} &nbsp;·&nbsp;
                    Probabilidad: <strong style="color:{color_nivel}">{prob*100:.1f}%</strong>
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("##### Datos del estudiante")
        datos = {
            "📊 Promedio de notas":  f"{est['promedio_notas']:.2f} / 20",
            "🔢 Nota matemática":    f"{est['nota_matematica']:.2f} / 20",
            "📝 Nota comunicación":  f"{est['nota_comunicacion']:.2f} / 20",
            "📅 Asistencia":         f"{est['porcentaje_asistencia']:.1f} %",
            "🤝 Conducta":           f"{int(est['nivel_conducta'])} / 5",
            "🙋 Participación":      f"{int(est['nivel_participacion'])} / 5",
            "📈 Tendencia de notas": tendencia_txt,
        }
        st.dataframe(
            pd.DataFrame({"Variable": list(datos.keys()), "Valor": list(datos.values())}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 46, "color": color_nivel}},
            title={"text": "Probabilidad de riesgo", "font": {"size": 14, "color": "#64748b"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar":  {"color": color_nivel, "thickness": 0.28},
                "bgcolor": "#f8fafc",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  45], "color": "#dcfce7"},
                    {"range": [45, 70], "color": "#fef9c3"},
                    {"range": [70, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": color_nivel, "width": 4},
                    "thickness": 0.85,
                    "value": prob * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            height=310,
            margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_family="Inter, sans-serif",
            font_color="#1e293b",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Top 3 factores
    importancias = metrics.get("feature_importances") or []
    if not importancias:
        return

    df_imp = (
        pd.DataFrame({"feature": FEATURES, "importancia": importancias})
        .sort_values("importancia", ascending=False)
    )
    top3 = df_imp.head(3)["feature"].tolist()

    st.markdown("##### Factores con mayor peso en esta predicción")
    chip_cols = st.columns(3)
    for i, feat in enumerate(top3):
        valor_txt = _valor_texto(feat, est[feat], tendencia_txt)
        imp_val   = df_imp[df_imp["feature"] == feat]["importancia"].values[0]
        imp_pct   = imp_val * 100
        bar_w     = min(imp_pct * 5, 100)
        icon      = FEATURE_ICONS.get(feat, "📌")

        with chip_cols[i]:
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:18px 20px;
                 border-top:3px solid {color_nivel};
                 box-shadow:0 1px 4px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);
                 height:148px">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:10px">
                    <span style="font-size:15px">{icon}</span>
                    <span style="font-size:10px;color:#94a3b8;font-weight:700;
                         text-transform:uppercase;letter-spacing:.8px">
                        #{i+1} · {FEATURE_LABELS[feat]}
                    </span>
                </div>
                <div style="font-size:30px;font-weight:700;color:#0f172a">{valor_txt}</div>
                <div style="margin-top:10px;background:#f1f5f9;border-radius:4px;height:4px">
                    <div style="height:4px;width:{bar_w:.0f}%;
                         background:{color_nivel};border-radius:4px"></div>
                </div>
                <div style="font-size:11px;color:#94a3b8;margin-top:4px">
                    Importancia: {imp_pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    explicacion = construir_explicacion(est, top3, nivel, prob, tendencia_txt)
    bg = {"ALTO": "#FEF2F2", "MEDIO": "#FFFBEB", "BAJO": "#F0FDF4"}[nivel]
    st.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:18px 22px;
         border-left:4px solid {color_nivel};margin-top:16px">
        <div style="font-size:11px;font-weight:700;color:{color_nivel};
             text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px">
            💡 Explicación del riesgo
        </div>
        <div style="color:#1e293b;font-size:14px;line-height:1.6">{explicacion}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────

def render_footer(metrics: dict) -> None:
    items = [
        ("Accuracy",  metrics.get("accuracy", 0)),
        ("Precision", metrics.get("precision", 0)),
        ("Recall",    metrics.get("recall", 0)),
        ("F1-Score",  metrics.get("f1_score", 0)),
        ("AUC-ROC",   metrics.get("auc_roc", 0)),
    ]
    badges = "".join([
        f'<div style="text-align:center">'
        f'<div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px">{n}</div>'
        f'<div style="font-size:18px;font-weight:700;color:#0f172a">{v:.3f}</div>'
        f'</div>'
        for n, v in items
    ])
    st.markdown(f"""
    <div style="background:#f8fafc;border-radius:12px;padding:16px 24px;margin-top:16px;
         border:1px solid #e2e8f0;display:flex;gap:32px;flex-wrap:wrap;align-items:center">
        <div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:1px;
             text-transform:uppercase;white-space:nowrap">Métricas del modelo (test)</div>
        {badges}
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not verificar_archivos():
        st.stop()

    model, encoder, metrics = cargar_modelo()
    df      = cargar_dataset()
    df_pred = generar_predicciones(df, model, encoder)
    filtros = render_sidebar(df_pred, metrics)
    df_filt = aplicar_filtros(df_pred, filtros)

    render_header(metrics)
    seccion_kpis(df_filt)
    st.divider()
    seccion_ranking(df_filt)
    st.divider()
    seccion_visualizaciones(df_filt, metrics)
    st.divider()
    seccion_detalle_estudiante(df_filt, metrics)
    render_footer(metrics)


if __name__ == "__main__":
    main()
