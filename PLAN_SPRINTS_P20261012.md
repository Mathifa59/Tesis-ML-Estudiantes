# Plan de Sprints — Proyecto P20261012
## Sistema de Alerta Temprana de Riesgo Académico
**UPC · Taller de Proyectos I · Ingeniería de Sistemas de Información**

---

## 1. Información general

| Campo | Detalle |
|---|---|
| **Código del proyecto** | P20261012 |
| **Curso** | Taller de Proyectos I |
| **Carrera** | Ingeniería de Sistemas de Información |
| **Project Manager** | Torres Saldaña, Gabriel Alonso |
| **Scrum Manager** | Tong Barahona, Dylan |
| **Product Owner** | Jose Luis Santisteban Pazos |
| **Total de HUs** | 35 historias de usuario · 73 escenarios |
| **Épicas** | 7 (EP01 — EP07) |
| **Roles del sistema** | Administrador del Sistema · Director / Coordinador Académico |

---

## 2. Estructura de Sprints

| Sprint | Semanas UPC | Dailies | Foco temático |
|---|---|---|---|
| **Sprint 1** | Sem 4, 5, 6 | 9 daily scrums (3 por semana) | Fundamentos · Modelo ML · Diseño de arquitectura |
| **Sprint 2** | Sem 7, 8, 9 | 9 daily scrums | Backend ML (FastAPI) + Backend Principal (Supabase) |
| **Sprint 3** | Sem 10, 11, 12 | 9 daily scrums | Frontend Next.js + Vistas core (Director y Admin) |
| **Sprint 4** | Sem 13, 14, 15 | 9 daily scrums | Features avanzadas + Despliegue + Sustentación |

**Cadencia:** 3 daily scrums por semana × 3 semanas = 9 dailies por sprint.

---

## 3. Arquitectura objetivo (resumen)

| Capa | Stack | Responsabilidad |
|---|---|---|
| **Frontend** | Next.js 14 · TypeScript · Tailwind · shadcn/ui · Plotly.js | Vistas Admin y Director, intervenciones, reportes, visualizaciones |
| **Backend Principal** | Supabase (Auth, Storage, Edge Functions, Realtime) | Autenticación con roles, almacenamiento, lógica de negocio, notificaciones |
| **Backend ML** | FastAPI (Python) | Endpoints `/predecir`, `/reentrenar`, `/metricas`, `/importancia` |
| **Capa IA** | scikit-learn (Random Forest) + imbalanced-learn (SMOTE) | Modelo predictivo, validación cruzada, feature importance, versionado |
| **Base de Datos** | Postgres (Supabase) | 7 tablas: profiles, estudiantes, notas_periodos, predicciones, intervenciones, modelos_versiones, audit_log |
| **Cloud** | Vercel · Supabase Cloud · Railway/Render · Resend · GitHub Actions · Sentry | Despliegue, CI/CD, monitoreo, notificaciones email |

Referencia visual: [Arquitectura_P20261012.drawio](Arquitectura_P20261012.drawio) y [Arquitectura_Fisica_P20261012.drawio](Arquitectura_Fisica_P20261012.drawio).

---

## 4. Catálogo de Épicas y HUs

| Épica | Nombre | HUs incluidas |
|---|---|---|
| **EP01** | Acceso y Seguridad | HU001, HU002, HU003, HU004, HU005 |
| **EP02** | Predicción y Clasificación de Riesgo | HU006, HU007, HU008, HU009, HU010 |
| **EP03** | Análisis y Visualización de Estudiantes | HU011, HU012, HU013, HU014, HU015, HU016, HU017 |
| **EP04** | Priorización e Intervención | HU018, HU019, HU020, HU021, HU022 |
| **EP05** | Seguimiento Histórico y Reportes | HU023, HU024, HU025, HU026 |
| **EP06** | Gestión de Datos | HU027, HU028, HU029, HU030, HU031 |
| **EP07** | Mantenimiento del Modelo ML | HU032, HU033, HU034, HU035 |

---

# 5. SPRINT 1 — Semanas 4, 5, 6

## Sprint Goal
> *Tener el modelo Random Forest entrenado, validado y documentado; el dataset definido y normalizado; y la arquitectura lógica/física + esquema de base de datos diseñados y aprobados, dejando las bases listas para iniciar el desarrollo de back y front.*

## HUs comprometidas

| HU | Título | Épica | Justificación |
|---|---|---|---|
| **HU010** | Ejecutar el modelo predictivo con un solo botón | EP02 | Núcleo del modelo ML que se entrena en este sprint |
| **HU032** | Ajustar parámetros del modelo predictivo | EP07 | Tuning de hiperparámetros (n_estimators, max_depth) |
| **HU033** | Entrenar el modelo con nuevos datos institucionales | EP07 | Pipeline de entrenamiento con `train_model.py` |
| **HU034** | Visualizar importancia global de variables | EP07 | Feature importance generado y persistido |
| **HU035** | Gestionar fecha de última actualización del modelo | EP07 | Timestamp guardado en `metricas_modelo.pkl` |

## Detalle por daily

| Sem | Daily | Fecha tentativa | Tarea | HU vinculada |
|---|---|---|---|---|
| **Sem 4** | D1 | Lunes Sem 4 | Kickoff, definición de roles, repositorio Git, convenciones | — |
| | D2 | Miércoles Sem 4 | Definición del problema, KPIs, criterios de éxito | — |
| | D3 | Viernes Sem 4 | Diseño del dataset (10 features + target, escala vigesimal) | HU033 |
| **Sem 5** | D4 | Lunes Sem 5 | Generación de `dataset_estudiantes.csv` (300 registros) | HU033 |
| | D5 | Miércoles Sem 5 | EDA en notebook (distribuciones, correlaciones, balance de clases) | HU033 |
| | D6 | Viernes Sem 5 | Entrenamiento comparativo: Random Forest vs Logistic Regression vs Decision Tree | HU032, HU033 |
| **Sem 6** | D7 | Lunes Sem 6 | SMOTE + tuning RF (n_estimators=200, max_depth=8, min_samples_split=5) | HU032 |
| | D8 | Miércoles Sem 6 | Validación cruzada 5-fold + métricas finales + feature importance | HU010, HU034, HU035 |
| | D9 | Viernes Sem 6 | Diseño esquema Postgres (7 tablas) + arquitectura lógica + arquitectura física | — |

## Entregables del Sprint 1
- [dataset_estudiantes.csv](dataset_estudiantes.csv) con 300 registros validados
- [modelo_riesgo_academico.ipynb](modelo_riesgo_academico.ipynb) — notebook EDA y comparación
- [train_model.py](train_model.py) — pipeline de entrenamiento automatizado
- [modelo_rf.pkl](modelo_rf.pkl), [label_encoder.pkl](label_encoder.pkl), [metricas_modelo.pkl](metricas_modelo.pkl)
- [Arquitectura_P20261012.drawio](Arquitectura_P20261012.drawio) — arquitectura lógica
- [Arquitectura_Fisica_P20261012.drawio](Arquitectura_Fisica_P20261012.drawio) — arquitectura física
- Esquema SQL inicial de las 7 tablas

## Definition of Done — Sprint 1
- Modelo entrena en menos de 10 segundos con `python train_model.py`
- Métricas finales: Accuracy ≥ 0.80, Recall ≥ 0.70, AUC-ROC ≥ 0.85
- Validación cruzada 5-fold ejecutada y reportada
- SMOTE aplicado únicamente al set de entrenamiento (nunca al de test)
- Esquema de BD documentado con tipos de datos, claves y relaciones

---

# 6. SPRINT 2 — Semanas 7, 8, 9

## Sprint Goal
> *Levantar las dos capas de backend completamente funcionales: el servicio ML expuesto vía REST (FastAPI) y el backend principal con base de datos, autenticación y storage (Supabase), incluyendo carga masiva de datos.*

## HUs comprometidas

| HU | Título | Épica | Capa |
|---|---|---|---|
| **HU001** | Iniciar sesión con credenciales | EP01 | Auth |
| **HU002** | Cerrar sesión de forma segura | EP01 | Auth |
| **HU003** | Recuperar contraseña en caso de olvido | EP01 | Auth |
| **HU004** | Gestionar usuarios del sistema (alta, baja, roles) | EP01 | Auth |
| **HU005** | Auditar accesos al sistema | EP01 | DB + audit_log |
| **HU006** | Identificar estudiantes con riesgo de bajo rendimiento | EP02 | Endpoint `/predecir` |
| **HU009** | Automatizar el análisis del rendimiento estudiantil | EP02 | Edge Function + cron |
| **HU027** | Integrar datos de distintas fuentes institucionales | EP06 | Edge Function |
| **HU028** | Limpiar y validar los datos automáticamente | EP06 | Edge Function de validación |
| **HU029** | Actualizar periódicamente la base de datos | EP06 | Cron job en Edge Functions |
| **HU030** | Cargar datos al sistema mediante archivos CSV/XLSX | EP06 | Supabase Storage + Edge Function |
| **HU031** | Visualizar errores en datos cargados | EP06 | Endpoint de validación |

## Detalle por daily

| Sem | Daily | Fecha tentativa | Tarea | HU vinculada |
|---|---|---|---|---|
| **Sem 7** | D1 | Lunes Sem 7 | Setup proyecto FastAPI, estructura, carga del `.pkl` en memoria | HU010 (carry over) |
| | D2 | Miércoles Sem 7 | Endpoint `POST /predecir` (batch de estudiantes) | HU006 |
| | D3 | Viernes Sem 7 | Endpoints `GET /metricas` y `GET /importancia` | HU034, HU035 |
| **Sem 8** | D4 | Lunes Sem 8 | Endpoint `POST /reentrenar` + versionado de modelos | HU033 |
| | D5 | Miércoles Sem 8 | Setup Supabase, migraciones SQL de las 7 tablas | — |
| | D6 | Viernes Sem 8 | Supabase Auth + roles (admin/director) + Row Level Security | HU001, HU002, HU003, HU004 |
| **Sem 9** | D7 | Lunes Sem 9 | Supabase Storage (CSV uploads, .pkl, reportes PDF) | HU030 |
| | D8 | Miércoles Sem 9 | Edge Functions: validación de schema, cron de actualización, audit_log | HU005, HU027, HU028, HU029, HU031 |
| | D9 | Viernes Sem 9 | Integración FastAPI ↔ Supabase + tests de endpoints + Edge Function automatización | HU009 |

## Entregables del Sprint 2
- API FastAPI desplegada localmente con 4 endpoints funcionales y documentados (Swagger)
- Base de datos Postgres con migraciones aplicadas (7 tablas)
- Supabase Auth funcionando con 2 roles y RLS activo
- Carga masiva de CSV operativa (con validación y reporte de errores)
- Edge Functions desplegadas para validación, cron y notificaciones
- Tests de integración pasando (mínimo 80% de cobertura en endpoints críticos)

## Definition of Done — Sprint 2
- Endpoint `/predecir` responde en menos de 2 segundos para 300 estudiantes
- Login y logout funcionan con JWT y expiran a los 30 minutos de inactividad (HU002)
- RLS bloquea efectivamente accesos cruzados entre roles
- Carga de CSV inválido devuelve mensaje específico de error por fila (HU031)
- Cron de actualización corre y queda registrado en `audit_log` (HU005, HU029)

---

# 7. SPRINT 3 — Semanas 10, 11, 12

## Sprint Goal
> *Construir el frontend Next.js completo con autenticación, vista Director (KPIs, ranking, ficha de estudiante con factores y explicación) y vista Admin (dashboard del modelo, importancia, ROC, matriz de confusión), conectado al backend en tiempo real.*

## HUs comprometidas

| HU | Título | Épica |
|---|---|---|
| **HU007** | Visualizar el nivel de riesgo de los estudiantes | EP02 |
| **HU008** | Clasificar estudiantes según nivel de riesgo | EP02 |
| **HU011** | Conocer los factores específicos que elevan el riesgo | EP03 |
| **HU012** | Visualizar indicadores globales del colegio (KPIs) | EP03 |
| **HU013** | Ver ranking de estudiantes según nivel de riesgo | EP03 |
| **HU014** | Filtrar estudiantes por grado o sección | EP03 |
| **HU015** | Visualizar la distribución de estudiantes por nivel | EP03 |
| **HU016** | Ver explicación simple del riesgo por estudiante | EP03 |
| **HU017** | Ver historial académico del estudiante en un solo lugar | EP03 |
| **HU020** | Priorizar estudiantes según urgencia de intervención | EP04 |
| **HU021** | Segmentar estudiantes por tipo de riesgo | EP04 |

## Detalle por daily

| Sem | Daily | Fecha tentativa | Tarea | HU vinculada |
|---|---|---|---|---|
| **Sem 10** | D1 | Lunes Sem 10 | Setup Next.js 14 + Tailwind + shadcn/ui + sistema de diseño | — |
| | D2 | Miércoles Sem 10 | Pantalla Login + integración Supabase Auth + protección de rutas | HU001, HU002, HU003 (UI) |
| | D3 | Viernes Sem 10 | Layout base, sidebar oscuro, header con métricas del modelo | HU035 |
| **Sem 11** | D4 | Lunes Sem 11 | Vista Director: KPIs globales (4 tarjetas Total/Alto/Medio/Bajo) | HU012, HU015 |
| | D5 | Miércoles Sem 11 | Vista Director: Ranking por urgencia + filtros (grado/sección/nivel) | HU007, HU008, HU013, HU014, HU020, HU021 |
| | D6 | Viernes Sem 11 | Vista Director: Ficha del estudiante (gauge + top 3 factores + explicación) | HU011, HU016, HU017 |
| **Sem 12** | D7 | Lunes Sem 12 | Vista Admin: Dashboard del modelo (Accuracy, F1, AUC, fecha actualización) | HU034, HU035 |
| | D8 | Miércoles Sem 12 | Visualizaciones Plotly.js (distribución, importancia, matriz de confusión, ROC) | HU015, HU034 |
| | D9 | Viernes Sem 12 | Realtime: WebSocket Supabase para actualizaciones en vivo + UI de panel admin de usuarios | HU004 (UI) |

## Entregables del Sprint 3
- Aplicación Next.js desplegada localmente con login funcional
- Vista Director completa (KPIs, ranking, filtros, ficha)
- Vista Administrador completa (gestión de usuarios, dashboard del modelo)
- Visualizaciones Plotly.js integradas y responsivas
- Conexión Realtime activa (cambios en BD se reflejan sin refresh)

## Definition of Done — Sprint 3
- Tiempo de carga inicial menor a 3 segundos
- 100% responsive (mobile, tablet, desktop) — breakpoints validados
- Cumple estándar de accesibilidad WCAG 2.1 nivel AA en componentes críticos
- Filtros aplican en menos de 200 ms sobre 300 estudiantes
- Estados de carga, error y vacío implementados en cada pantalla

---

# 8. SPRINT 4 — Semanas 13, 14, 15

## Sprint Goal
> *Cerrar las funcionalidades avanzadas (intervenciones, alertas email, reportes históricos en PDF), desplegar el sistema completo a la nube (Vercel + Supabase Cloud + Railway), establecer CI/CD y monitoreo, y preparar la sustentación final.*

## HUs comprometidas

| HU | Título | Épica |
|---|---|---|
| **HU018** | Recibir alertas de estudiantes en riesgo | EP04 |
| **HU019** | Recibir recomendaciones de acciones de intervención | EP04 |
| **HU022** | Registrar las intervenciones realizadas | EP04 |
| **HU023** | Monitorear el riesgo académico en el tiempo | EP05 |
| **HU024** | Generar reportes históricos del rendimiento estudiantil | EP05 |
| **HU025** | Comparar el riesgo entre periodos académicos | EP05 |
| **HU026** | Exportar listados de estudiantes en riesgo (CSV/PDF) | EP05 |

## Detalle por daily

| Sem | Daily | Fecha tentativa | Tarea | HU vinculada |
|---|---|---|---|---|
| **Sem 13** | D1 | Lunes Sem 13 | Módulo de Registro de Intervenciones (tutorías, reuniones, derivaciones) | HU022 |
| | D2 | Miércoles Sem 13 | Motor de recomendaciones de intervención + UI de visualización | HU019 |
| | D3 | Viernes Sem 13 | Notificaciones email (Resend/SendGrid) para riesgo ALTO + alertas in-app | HU018 |
| **Sem 14** | D4 | Lunes Sem 14 | Reportes históricos en PDF + comparación entre periodos | HU023, HU024, HU025 |
| | D5 | Miércoles Sem 14 | Exportación de listados (CSV / PDF) desde el frontend | HU026 |
| | D6 | Viernes Sem 14 | Despliegue Frontend a Vercel + FastAPI a Railway/Render + Supabase Cloud producción | — |
| **Sem 15** | D7 | Lunes Sem 15 | CI/CD con GitHub Actions + monitoreo Sentry (frontend + backend) | — |
| | D8 | Miércoles Sem 15 | QA integral, pruebas de usuario, ajustes finales, documentación | — |
| | D9 | Viernes Sem 15 | Preparación de sustentación: demo, slides, análisis de cumplimiento de HUs | — |

## Entregables del Sprint 4
- Sistema completo desplegado en producción (URLs públicas Vercel + Railway)
- Notificaciones email funcionando para estudiantes en riesgo ALTO
- Generación de PDFs y CSVs operativa
- Pipeline de CI/CD activo (push a `main` → deploy automático)
- Sentry capturando errores en producción
- Documentación final + análisis de cumplimiento de HUs ([Analisis_Cumplimiento_HUs_P20261012.docx](Analisis_Cumplimiento_HUs_P20261012.docx))
- Material de sustentación (slides + demo grabada de respaldo)

## Definition of Done — Sprint 4
- 35/35 HUs completadas y validadas con sus criterios de aceptación
- Despliegue accesible desde URL pública con HTTPS
- Notificación email llega en menos de 1 minuto tras detección de riesgo ALTO
- PDFs generados respetan branding del proyecto
- Cero errores críticos en Sentry durante la semana de sustentación

---

## 9. Mapeo completo: HU → Sprint

| HU | Título corto | Épica | Sprint asignado |
|---|---|---|---|
| HU001 | Iniciar sesión | EP01 | **Sprint 2** (back) + Sprint 3 (UI) |
| HU002 | Cerrar sesión segura | EP01 | **Sprint 2** (back) + Sprint 3 (UI) |
| HU003 | Recuperar contraseña | EP01 | **Sprint 2** (back) + Sprint 3 (UI) |
| HU004 | Gestionar usuarios | EP01 | **Sprint 2** (back) + Sprint 3 (UI) |
| HU005 | Auditar accesos | EP01 | **Sprint 2** |
| HU006 | Identificar estudiantes en riesgo | EP02 | **Sprint 2** |
| HU007 | Visualizar nivel de riesgo | EP02 | **Sprint 3** |
| HU008 | Clasificar según nivel | EP02 | **Sprint 3** |
| HU009 | Automatizar análisis | EP02 | **Sprint 2** |
| HU010 | Ejecutar modelo con un botón | EP02 | **Sprint 1** |
| HU011 | Conocer factores de riesgo | EP03 | **Sprint 3** |
| HU012 | Indicadores globales (KPIs) | EP03 | **Sprint 3** |
| HU013 | Ranking de estudiantes | EP03 | **Sprint 3** |
| HU014 | Filtrar por grado/sección | EP03 | **Sprint 3** |
| HU015 | Distribución por nivel | EP03 | **Sprint 3** |
| HU016 | Explicación simple del riesgo | EP03 | **Sprint 3** |
| HU017 | Historial académico | EP03 | **Sprint 3** |
| HU018 | Alertas de riesgo | EP04 | **Sprint 4** |
| HU019 | Recomendaciones de intervención | EP04 | **Sprint 4** |
| HU020 | Priorizar por urgencia | EP04 | **Sprint 3** |
| HU021 | Segmentar por tipo de riesgo | EP04 | **Sprint 3** |
| HU022 | Registrar intervenciones | EP04 | **Sprint 4** |
| HU023 | Monitoreo histórico | EP05 | **Sprint 4** |
| HU024 | Reportes históricos PDF | EP05 | **Sprint 4** |
| HU025 | Comparar entre periodos | EP05 | **Sprint 4** |
| HU026 | Exportar listados | EP05 | **Sprint 4** |
| HU027 | Integrar fuentes externas | EP06 | **Sprint 2** |
| HU028 | Limpieza y validación de datos | EP06 | **Sprint 2** |
| HU029 | Actualización periódica de BD | EP06 | **Sprint 2** |
| HU030 | Carga de datos por archivo | EP06 | **Sprint 2** |
| HU031 | Visualizar errores de carga | EP06 | **Sprint 2** |
| HU032 | Ajustar parámetros del modelo | EP07 | **Sprint 1** |
| HU033 | Entrenar con nuevos datos | EP07 | **Sprint 1** + Sprint 2 (endpoint) |
| HU034 | Importancia global de variables | EP07 | **Sprint 1** + Sprint 3 (UI) |
| HU035 | Fecha de última actualización | EP07 | **Sprint 1** + Sprint 3 (UI) |

---

## 10. Distribución de carga por sprint

| Sprint | HUs principales | HUs colaborativas | % del backlog |
|---|---|---|---|
| Sprint 1 | 5 (HU010, HU032–HU035) | — | 14 % |
| Sprint 2 | 12 (HU001–HU006, HU009, HU027–HU031) | HU033 (endpoint) | 34 % |
| Sprint 3 | 11 (HU007, HU008, HU011–HU017, HU020, HU021) | HU001–HU004 (UI), HU034, HU035 | 31 % |
| Sprint 4 | 7 (HU018, HU019, HU022, HU023–HU026) | — | 21 % |
| **Total** | **35 HUs** | — | **100 %** |

---

## 11. Eventos Scrum por sprint

| Evento | Frecuencia | Duración | Notas |
|---|---|---|---|
| **Sprint Planning** | 1 vez al inicio de cada sprint | 1 — 2 horas | Selección de HUs y estimación |
| **Daily Scrum** | 3 veces por semana | 15 minutos | ¿Qué hice? ¿Qué haré? ¿Bloqueos? |
| **Sprint Review** | 1 vez al final de cada sprint | 1 hora | Demo al Product Owner |
| **Sprint Retrospective** | 1 vez al final de cada sprint | 30 — 45 minutos | Mejoras de proceso |

**Total de daily scrums por sprint:** 9
**Total de daily scrums en el proyecto:** 36

---

## 12. Riesgos identificados y mitigación

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Dataset simulado no refleja patrones reales | Media | Alto | Documentar claramente la limitación; coordinar con colegio piloto si es viable |
| Falsos negativos altos (estudiantes en riesgo no detectados) | Media | Alto | Priorizar Recall sobre Accuracy; ajustar umbral de probabilidad |
| Despliegue en cloud falla cerca de la sustentación | Baja | Alto | Hacer deploys de prueba desde Sprint 3; tener demo local de respaldo |
| Atraso en backend bloquea el frontend del Sprint 3 | Media | Medio | Definir contratos de API en Sprint 1; usar mocks en frontend si es necesario |
| Notificaciones email no llegan (filtros antispam) | Media | Bajo | Configurar SPF/DKIM; tener notificación in-app como respaldo |
| Sobrecarga de HUs en Sprint 2 (12 HUs) | Alta | Medio | Considerar mover HU027 (integración fuentes externas) a Sprint 4 si hay riesgo |

---

## 13. Métricas y KPIs del modelo (objetivo)

| Métrica | Objetivo | Justificación |
|---|---|---|
| **Accuracy** | ≥ 0.80 | Indicador general de aciertos |
| **Precision** | ≥ 0.70 | Reducir falsos positivos (alertas innecesarias) |
| **Recall** | ≥ 0.70 | **Métrica crítica** — minimizar falsos negativos |
| **F1-Score** | ≥ 0.70 | Balance entre Precision y Recall |
| **AUC-ROC** | ≥ 0.85 | Calidad global del clasificador |
| **CV F1 (5-fold)** | ≥ 0.65 ± 0.10 | Estabilidad del modelo |

**Niveles de riesgo:**
- 🔴 **ALTO** — probabilidad ≥ 0.70 → intervención inmediata
- 🟡 **MEDIO** — probabilidad 0.45 – 0.69 → monitoreo cercano
- 🟢 **BAJO** — probabilidad < 0.45 → sin alerta

---

## 14. Limitaciones declaradas

- Dataset de **300 estudiantes simulados** (no refleja la diversidad de un colegio real).
- No incluye **factores socioemocionales** (bullying, salud mental, situación familiar).
- Sin **integración en tiempo real** con sistemas académicos preexistentes.
- Contexto restringido a **colegios privados de Lima Metropolitana**.
- El modelo es un MVP académico — para producción real requeriría validación con un colegio piloto y datos longitudinales.

---

*Documento generado para el proyecto **P20261012** · UPC · Taller de Proyectos I*
*Última actualización: mayo 2026*
