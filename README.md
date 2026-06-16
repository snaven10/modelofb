# ⚽ modelofb — Predicción de partidos del Mundial 2026

Proyecto **personal de aprendizaje**: construir, paso a paso, un sistema que
prediga **probabilidades bien calibradas** de resultados de fútbol (victoria
local / empate / victoria visitante, y marcadores), aplicado al **Mundial 2026**.

> El objetivo NO es "adivinar el marcador exacto". Eso es casi imposible y de poco
> valor. El objetivo es estimar **probabilidades realistas** (ej. "Brasil 58% gana,
> 24% empata, 18% pierde") y que esas probabilidades estén **calibradas**: cuando
> el modelo dice 60%, que efectivamente pase ~60% de las veces.

## ¿Qué hace?

1. **Carga datos** de varias fuentes (históricos con xG, ligas con odds, Mundial real).
2. **Construye features** (forma reciente, xG acumulado, localía, ranking).
3. **Entrena modelos** de menor a mayor sofisticación:
   - **Elo** — ratings simples, el "hola mundo" de la predicción deportiva.
   - **Poisson / Dixon-Coles** — el núcleo: modela goles esperados por equipo.
   - **Poisson con xG** — misma idea, pero alimentado con xG en vez de goles crudos.
   - **(Opcional) ML** — regresión logística / gradient boosting con features.
4. **Evalúa** con log-loss, Brier score, accuracy y curvas de calibración, y hace
   **backtesting** contra las odds de cierre del mercado.
5. **Predice** los partidos reales del Mundial 2026 con el mejor modelo.

## Fuentes de datos

| Fuente | Para qué | API key |
|--------|----------|---------|
| **StatsBomb Open Data** (`statsbombpy`) | Históricos con xG ya calculado. Mundial M 2022 (`comp 43`) y F 2023 (`comp 72`). | ❌ No |
| **football-data.co.uk** | Ligas históricas con resultados + odds de cierre (entrenar y calibrar). | ❌ No |
| **martj42/international_results** | ~49k partidos de SELECCIONES (1872→hoy): amistosos, eliminatorias, continentales y Mundiales. El insumo del Elo de selecciones. Trae `neutral` y `tournament`. | ❌ No |
| **API-Football** (API-SPORTS) | Partidos reales del Mundial 2026 (`league=1`, `season=2026`). Free: 100 req/día → **caché obligatoria**. | ✅ Sí |
| **openfootball/worldcup.json** | (Opcional) histórico simple de Mundiales pasados. | ❌ No |

## Estructura

```
modelofb/
├── data/
│   ├── raw/          # datos crudos descargados (en .gitignore)
│   └── processed/    # datasets ya transformados (en .gitignore)
├── src/
│   ├── data_loaders/ # un loader por fuente de datos
│   ├── features/     # ingeniería de características
│   ├── models/       # Elo, Poisson/Dixon-Coles, xG, ML
│   ├── evaluation/   # métricas, calibración, backtesting
│   └── predict/      # aplicación final sobre el Mundial 2026
├── notebooks/        # exploración interactiva
├── tests/            # pruebas con pytest
├── requirements.txt
└── .env.example      # plantilla de configuración (copiá a .env)
```

## Instalación

> Requiere **Python 3.11+** (probado en 3.14).

```bash
# 1. Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# 2. Instalar dependencias
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Configurar secretos
cp .env.example .env
#    -> editá .env y pegá tu API_FOOTBALL_KEY (registrate gratis en api-football.com)
```

## Uso (lo que ya existe — Fase 1)

```bash
# Probar el loader de StatsBomb a mano (baja y cachea el Mundial M 2022):
python -m src.data_loaders.statsbomb_loader

# Correr las pruebas (no tocan la red, usan datos mockeados):
pytest -q
```

## Estado del proyecto

- [x] **Fase 1** — estructura, configuración y loader de StatsBomb.
- [~] **Fase 2** — loaders de datos:
  - [x] football-data.co.uk (CSVs con odds de cierre + probabilidades de mercado).
  - [x] descarga masiva top-5 europeas → `data/processed/matches_top5.parquet` (12,458 partidos).
  - [x] martj42/international_results (~49k partidos de selecciones, insumo del Elo).
  - [ ] API-Football (Mundial 2026, con caché en disco contra el límite diario).
- [~] **Fase 3** — modelos:
  - [x] Elo de selecciones (localía vía `neutral`, K modulado por `is_competitive`, 1X2 con empate modelado). Top-20 del ranking coincide con las potencias reales.
  - [x] Poisson (ataque/defensa + matriz de marcadores → 1X2, marcador exacto, over/under). Validado con Premier real.
  - [x] Dixon-Coles (corrección de marcadores bajos `ρ` + decay temporal `ξ`). DC+decay baja el gap vs mercado de +0.0788 a +0.0636; `ρ` casi no aporta en 1X2, `ξ` sí. **Campeón actual: +0.0590 con ξ=0.5 calibrado.**
  - [x] Ensemble Elo+Dixon-Coles (`src/models/ensemble.py`, log-pooling geométrico/aritmético, peso `w` calibrable). **Resultado honesto: NO superó a DC solo.** En validation w=0.5 se veía mejor (curva en U), pero en test intocado el ensemble dio gap +0.0619 vs +0.0590 de DC solo — el Elo de clubes es más débil (ignora goles) y el w no generalizó. Código queda como herramienta reutilizable.
  - [ ] Poisson con xG.
- [~] **Fase 4** — evaluación, calibración y backtesting:
  - [x] métricas: log-loss, Brier, accuracy, curva de calibración (`src/evaluation/metrics.py`).
  - [x] backtest temporal contra el mercado (`src/evaluation/backtest.py`). Poisson Premier: log-loss 1.0681 vs mercado 0.9893 (gap +0.0788; le gana al azar 1.0986, no al mercado).
  - [x] calibración de hiperparámetros con grid search y split train/val/test (`src/evaluation/calibrate.py`). Óptimo honesto en validation: `ρ=0.0`, `ξ=0.5` (curva en U). Veredicto en test intocado: gap **+0.0590** (Poisson base era +0.0788).
  - [ ] graficar curvas de fiabilidad.
- [ ] Fase 5 — aplicación final sobre el Mundial 2026.

## Filosofía

Código **claro y comentado**, que explica el **PORQUÉ** de cada decisión. Avanzamos
**por fases**, entendiendo los conceptos antes de tocar código. La AI ejecuta; vos
dirigís y entendés. Sin atajos.
