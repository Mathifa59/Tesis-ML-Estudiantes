# Plan de Ejecución — Proyecto P20261012
## Sistema de Alerta Temprana de Riesgo Académico
**UPC · Taller de Proyectos I · Ingeniería de Sistemas de Información**

---

## Decisiones técnicas cerradas

| Decisión | Elección | Razón |
|---|---|---|
| Estructura del repo | **Monorepo** | Un solo `git push`, CI compartido, más simple para 3 personas |
| Frontend | **Next.js 14 + TypeScript + Tailwind + shadcn/ui** | Estándar actual, deploy directo a Vercel |
| Backend principal | **Supabase** (Auth, Postgres, Storage, Realtime, Edge Functions) | Tier gratis cubre todo, auth + BD en un solo servicio |
| Backend ML | **FastAPI (Python)** | Único stack que puede cargar `.pkl` de scikit-learn |
| Deploy frontend | **Vercel** | Integración nativa con Next.js, preview por PR |
| Deploy backend ML | **Railway** | Soporta Dockerfile, tier gratis suficiente |
| Emails | **Resend** | API simple, tier gratis 100 emails/día |
| Monitoreo | **Sentry** | Gratis para proyectos pequeños, captura errores en producción |

---

## Equipo y división de trabajo

| Integrante | Rol Scrum | HUs asignadas | Épicas |
|---|---|---|---|
| **Gabriel Alonso Torres Saldaña** | Project Manager | HU027–HU035 | EP06 Gestión de Datos · EP07 Modelo ML |
| **Dylan Tong Barahona** | Scrum Manager | HU006–HU017 | EP02 Predicción · EP03 Análisis y Visualización |
| **Mathias (tú)** | Dev Lead | HU001–HU005 · HU018–HU026 | EP01 Auth · EP04 Intervención · EP05 Reportes |

**Regla de trabajo:** cada uno agarra una HU de punta a punta (migración SQL + endpoint + componente UI). No se divide por capas, se divide por funcionalidad.

---

## Cuentas necesarias (crear antes de arrancar)

| Servicio | Tier | URL | Para qué |
|---|---|---|---|
| GitHub | Free | github.com | Repo + CI/CD |
| Supabase | Free | supabase.com | BD + Auth + Storage + Realtime |
| Vercel | Free | vercel.com | Deploy frontend |
| Railway | Free | railway.app | Deploy FastAPI |
| Resend | Free | resend.com | Emails de alerta |
| Sentry | Free | sentry.io | Monitoreo de errores |

---

## Herramientas locales requeridas

```bash
node -v          # Necesitas Node.js 20 LTS
python --version # Necesitas Python 3.11+
npm install -g pnpm          # Gestor de paquetes para Next.js
npm install -g supabase      # Supabase CLI
```

VS Code + extensiones:
- **Tailwind CSS IntelliSense**
- **ESLint**
- **Prettier**
- **Python** (ms-python)
- **Thunder Client** (para testear endpoints FastAPI)

---

## Estructura del monorepo

```
Tesis-ML-Estudiantes/          ← repo actual (raíz del monorepo)
├── frontend/                  ← Next.js 14
│   ├── src/
│   │   ├── app/               ← App Router
│   │   │   ├── (auth)/        ← login, recuperar contraseña
│   │   │   ├── dashboard/     ← vista Director
│   │   │   └── admin/         ← vista Administrador
│   │   ├── components/        ← componentes shadcn/ui + propios
│   │   ├── lib/               ← supabase client, utils
│   │   └── types/             ← tipos TypeScript
│   ├── package.json
│   └── .env.local             ← variables de entorno (NO commitear)
│
├── backend-ml/                ← FastAPI
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── predict.py     ← POST /predecir
│   │   │   ├── metrics.py     ← GET /metricas
│   │   │   ├── retrain.py     ← POST /reentrenar
│   │   │   └── importance.py  ← GET /importancia
│   │   ├── models/            ← carga de .pkl
│   │   └── schemas.py         ← Pydantic models
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env                   ← variables de entorno (NO commitear)
│
├── supabase/
│   ├── migrations/            ← archivos SQL versionados
│   │   └── 0001_init.sql      ← 7 tablas iniciales
│   └── functions/             ← Edge Functions (Deno)
│       ├── trigger-alertas/
│       └── cron-actualizacion/
│
├── legacy-streamlit/          ← dashboard Streamlit actual (referencia)
│   ├── app.py
│   ├── train_model.py
│   └── ...
│
├── docs/                      ← documentación técnica
│   ├── PLAN_SPRINTS_P20261012.md
│   ├── PLAN_EJECUCION_P20261012.md  ← este archivo
│   └── Arquitectura_P20261012.drawio
│
├── .github/
│   └── workflows/
│       ├── frontend-ci.yml    ← lint + typecheck en cada PR
│       └── backend-ci.yml     ← ruff + tests en cada PR
│
├── .gitignore
└── README.md
```

---

## DÍA 1 — Setup del esqueleto (2-3 horas)

Ejecutar en este orden exacto. No saltar pasos.

### Paso 1 — Reorganizar el repo
```bash
# Desde la raíz del proyecto actual
mkdir legacy-streamlit frontend backend-ml supabase docs

# Mover archivos Streamlit a legacy
mv app.py train_model.py modelo_rf.pkl label_encoder.pkl \
   metricas_modelo.pkl dataset_estudiantes.csv \
   modelo_riesgo_academico.ipynb requirements.txt \
   listado_estudiantes_riesgo.csv legacy-streamlit/

# Mover documentación a docs
mv PLAN_SPRINTS_P20261012.md PLAN_EJECUCION_P20261012.md \
   DOCUMENTACION.md Arquitectura_P20261012.drawio \
   Arquitectura_Fisica_P20261012.drawio docs/
```

### Paso 2 — Inicializar Next.js
```bash
cd frontend
pnpm create next-app . --typescript --tailwind --app --src-dir --import-alias "@/*"
```

### Paso 3 — Instalar shadcn/ui
```bash
# Dentro de /frontend
pnpx shadcn@latest init
pnpx shadcn@latest add button card table badge dialog dropdown-menu select \
     skeleton toast progress avatar separator sheet
```

### Paso 4 — Inicializar FastAPI
```bash
# Desde /backend-ml
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn pydantic python-dotenv joblib \
            scikit-learn imbalanced-learn pandas numpy
pip freeze > requirements.txt
```

### Paso 5 — Inicializar Supabase
```bash
# Desde la raíz del repo
supabase init
supabase login
supabase link --project-ref TU_PROJECT_REF
```

### Paso 6 — Variables de entorno
`frontend/.env.local`:
```
NEXT_PUBLIC_SUPABASE_URL=https://xxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=xxxx
ML_API_URL=http://localhost:8000
```

`backend-ml/.env`:
```
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=xxxx
MODEL_PATH=../legacy-streamlit/modelo_rf.pkl
ENCODER_PATH=../legacy-streamlit/label_encoder.pkl
METRICS_PATH=../legacy-streamlit/metricas_modelo.pkl
```

### Paso 7 — Migración SQL inicial
Crear `supabase/migrations/0001_init.sql` con las 7 tablas:
```sql
-- profiles (usuarios + roles)
create table profiles (
  id uuid references auth.users primary key,
  email text not null,
  nombre text,
  rol text check (rol in ('admin', 'director')) default 'director',
  activo boolean default true,
  created_at timestamptz default now()
);

-- estudiantes
create table estudiantes (
  id uuid primary key default gen_random_uuid(),
  codigo text unique not null,
  nombre text,
  grado int,
  seccion text,
  created_at timestamptz default now()
);

-- notas_periodos
create table notas_periodos (
  id uuid primary key default gen_random_uuid(),
  estudiante_id uuid references estudiantes,
  periodo text,
  promedio_notas numeric,
  nota_matematica numeric,
  nota_comunicacion numeric,
  porcentaje_asistencia numeric,
  nivel_conducta int,
  nivel_participacion int,
  tendencia_notas int,
  created_at timestamptz default now()
);

-- predicciones
create table predicciones (
  id uuid primary key default gen_random_uuid(),
  estudiante_id uuid references estudiantes,
  probabilidad numeric,
  nivel text check (nivel in ('ALTO', 'MEDIO', 'BAJO')),
  modelo_version text,
  fecha timestamptz default now()
);

-- intervenciones
create table intervenciones (
  id uuid primary key default gen_random_uuid(),
  estudiante_id uuid references estudiantes,
  tipo text check (tipo in ('tutoria', 'reunion', 'derivacion')),
  descripcion text,
  registrado_por uuid references profiles,
  fecha timestamptz default now()
);

-- modelos_versiones
create table modelos_versiones (
  id uuid primary key default gen_random_uuid(),
  version text unique not null,
  accuracy numeric,
  precision_score numeric,
  recall numeric,
  f1 numeric,
  auc_roc numeric,
  storage_path text,
  activo boolean default false,
  created_at timestamptz default now()
);

-- audit_log
create table audit_log (
  id uuid primary key default gen_random_uuid(),
  usuario_id uuid references profiles,
  accion text,
  tabla text,
  detalle jsonb,
  ip text,
  created_at timestamptz default now()
);

-- Row Level Security
alter table profiles enable row level security;
alter table estudiantes enable row level security;
alter table notas_periodos enable row level security;
alter table predicciones enable row level security;
alter table intervenciones enable row level security;
alter table modelos_versiones enable row level security;
alter table audit_log enable row level security;
```

### Paso 8 — Verificar que todo levanta
```bash
# Terminal 1 — Frontend
cd frontend && pnpm dev        # → http://localhost:3000

# Terminal 2 — Backend ML
cd backend-ml
uvicorn app.main:app --reload  # → http://localhost:8000/docs
```

**Entregable del Día 1:** pantalla de login en `localhost:3000`, Swagger del FastAPI en `localhost:8000/docs`, Supabase con las 7 tablas creadas.

---

## SPRINT 1 (Sem 4–6) — Modelo ML listo, diseño y BD

| Daily | Tarea | Responsable |
|---|---|---|
| D1 | Reorganización del repo (Paso 1 de hoy) + README maestro | Todos |
| D2 | Migración SQL + Supabase Auth con roles | Mathias |
| D3 | FastAPI: estructura + carga del `.pkl` + Swagger | Dylan |
| D4 | Validar métricas del modelo actual (Acc 80%, AUC 0.866) | Gabriel |
| D5 | Next.js: layout base + sistema de colores + login UI | Mathias |
| D6 | FastAPI: `POST /predecir` batch completo | Dylan |
| D7 | Supabase: migrar dataset CSV a tabla `estudiantes` + `notas_periodos` | Gabriel |
| D8 | FastAPI: `GET /metricas`, `GET /importancia`, `POST /reentrenar` | Dylan |
| D9 | Integración frontend ↔ Supabase Auth + protección de rutas | Mathias |

---

## SPRINT 2 (Sem 7–9) — Backend completo + Auth funcionando

| Daily | Tarea | Responsable |
|---|---|---|
| D1 | Supabase Storage (CSV uploads) + Edge Function validación | Gabriel |
| D2 | HU001-HU003: Login, logout, recuperar contraseña | Mathias |
| D3 | HU004: Gestión de usuarios (admin puede crear/desactivar/cambiar rol) | Mathias |
| D4 | HU005: Panel de auditoría + audit_log trigger automático | Mathias |
| D5 | HU006: Predicción batch sobre todos los estudiantes (conecta FastAPI) | Dylan |
| D6 | HU009: Cron job — predicción automática periódica (Edge Function) | Dylan |
| D7 | HU030-HU031: Carga masiva de CSV con validación y reporte de errores | Gabriel |
| D8 | HU027-HU029: Integración de fuentes + limpieza + actualización periódica | Gabriel |
| D9 | Tests de integración: auth flow, `/predecir`, RLS, carga CSV | Todos |

---

## SPRINT 3 (Sem 10–12) — Frontend completo

| Daily | Tarea | Responsable |
|---|---|---|
| D1 | HU012: KPIs globales (4 tarjetas Total/Alto/Medio/Bajo) | Dylan |
| D2 | HU007-HU008: Visualización de niveles + clasificación | Dylan |
| D3 | HU013-HU014: Ranking + filtros por grado/sección/nivel | Dylan |
| D4 | HU011-HU016: Ficha del estudiante (gauge + factores + explicación) | Dylan |
| D5 | HU015: Visualizaciones Plotly.js (distribución, importancia, ROC, matriz) | Dylan |
| D6 | HU032-HU035: Vista Admin — configurar modelo, ver métricas, importancia | Gabriel |
| D7 | HU020-HU021: Ranking por urgencia + segmentación por tipo de riesgo | Mathias |
| D8 | HU017: Historial académico del estudiante | Dylan |
| D9 | Realtime: WebSocket Supabase — actualizaciones en vivo en dashboard | Mathias |

---

## SPRINT 4 (Sem 13–15) — Features avanzadas + Deploy + Sustentación

| Daily | Tarea | Responsable |
|---|---|---|
| D1 | HU022: Registro de intervenciones (tutorías, reuniones, derivaciones) | Mathias |
| D2 | HU018: Alertas email (Resend) cuando hay estudiante en riesgo ALTO | Mathias |
| D3 | HU019: Recomendaciones de intervención por tipo de riesgo | Mathias |
| D4 | HU023-HU025: Histórico de riesgo + comparación entre periodos | Gabriel |
| D5 | HU024-HU026: Reportes PDF + exportación CSV desde el frontend | Gabriel |
| D6 | Deploy: Vercel (frontend) + Railway (FastAPI) + Supabase Cloud (producción) | Todos |
| D7 | CI/CD: GitHub Actions (lint + typecheck + tests en cada PR) + Sentry | Dylan |
| D8 | QA integral: pruebas de usuario, ajustes de UX, corrección de bugs | Todos |
| D9 | Sustentación: demo grabada de respaldo + slides + análisis cumplimiento HUs | Todos |

---

## Flujo de trabajo en Git

```
main          ← producción (solo merge desde develop con PR aprobada)
develop       ← integración (se mergea al final de cada sprint)
feature/HU001-login
feature/HU006-prediccion-batch
feature/HU013-ranking-estudiantes
```

**Conventional commits:**
```
feat(HU001): implementar login con Supabase Auth
fix(HU013): corregir ordenamiento del ranking por probabilidad
docs: actualizar README con instrucciones de deploy
refactor(api): extraer lógica de predicción a servicio separado
```

**Regla de PR:**
- Título con el número de HU
- Screenshot o video del resultado
- Al menos 1 reviewer
- CI verde antes de merge

---

## Buenas prácticas — resumen

### TypeScript (frontend)
- `strict: true` en `tsconfig.json` — cero `any`
- Tipos en `src/types/` — nunca inline en componentes
- `zod` para validar formularios antes de enviar a Supabase

### Python (backend ML)
- Pydantic en todos los schemas de entrada y salida
- `ruff` como linter (reemplaza flake8 + isort + black)
- Variables de entorno solo por `.env` — nunca hardcodeadas

### Supabase
- Row Level Security en TODAS las tablas (ya está en el SQL inicial)
- Service key solo en backend-ml — nunca en el frontend
- `anon key` solo en frontend (es pública por diseño)

### Seguridad
- `.env.local` y `.env` en `.gitignore` — nunca commitear secrets
- JWT validado en cada endpoint de FastAPI con `python-jose`
- Nunca exponer el ID interno de Supabase en URLs públicas

---

## Checklist para validar que el Día 1 estuvo bien

- [ ] `pnpm dev` levanta sin errores en `localhost:3000`
- [ ] `uvicorn app.main:app --reload` levanta con Swagger en `localhost:8000/docs`
- [ ] Las 7 tablas existen en el dashboard de Supabase
- [ ] RLS está activado en todas las tablas
- [ ] `.env.local` y `.env` están en `.gitignore`
- [ ] El primer commit al repo tiene la nueva estructura de monorepo
- [ ] `legacy-streamlit/app.py` sigue corriendo (`streamlit run legacy-streamlit/app.py`)

---

## Lo que NO hacemos (para no perder tiempo)

- No reescribir el modelo ML — el `.pkl` actual cumple métricas, solo lo exponemos vía API
- No usar Redux/Zustand — React Server Components + hooks de Supabase son suficientes
- No escribir tests para todo — solo auth flow, `/predecir`, RLS y carga CSV
- No diseñar UI desde cero — copiar y adaptar componentes de shadcn/ui
- No hacer deploy hasta el Sprint 4 — primero que funcione local

---

*Última actualización: mayo 2026 · Proyecto P20261012 · UPC*
