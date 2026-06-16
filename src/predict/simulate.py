"""
simulate.py — Simulación Monte Carlo del Mundial 2026.

LA IDEA (propagar incertidumbre):
    Un partido no es determinista: aunque España sea favorita, a veces pierde. Para
    saber "¿qué probabilidad tiene cada selección de avanzar / ser campeón?" no basta
    con predecir partidos sueltos: hay que JUGAR EL TORNEO ENTERO miles de veces, cada
    vez con un dado cargado por las probabilidades del modelo, y contar en qué fracción
    de esos universos cada equipo llega a cada instancia.

MODELO DE GOLES:
    Simulamos MARCADORES con un Poisson de selecciones (entrenado desde 2010). Por qué
    goles y no solo 1X2: la fase de grupos se desempata por DIFERENCIA DE GOL y GOLES A
    FAVOR, así que necesitamos marcadores, no solo quién gana. Y en eliminatorias un
    empate va a prórroga/penales, que también se resuelve con goles.

CANCHA NEUTRAL (lo que insistimos desde el día uno):
    El Poisson tiene la localía horneada en sus medias (home_mean > away_mean). En el
    Mundial casi todo es cancha neutral, así que para esos partidos PROMEDIAMOS las dos
    orientaciones (X-local⊕Y-local): el sesgo de "quién está en casa" se cancela. Para
    los anfitriones (neutral=False) usamos la orientación normal, con su localía.

Este módulo cubre la FASE DE GRUPOS (12 grupos de 4, round-robin real). El cuadro de
eliminatorias se ensambla aparte (bracket.py) sobre estas mismas primitivas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.data_loaders.international_loader import InternationalResultsLoader
from src.models.poisson import PoissonModel

logger = logging.getLogger(__name__)

WORLD_CUP_2026_FROM = "2026-06-01"


# ──────────────────────────────────────────────────────────────────────────────
# Primitiva: λ neutralizado + muestreo de un marcador
# ──────────────────────────────────────────────────────────────────────────────
def expected_goals(model: PoissonModel, a: str, b: str,
                   neutral: bool) -> tuple[float, float]:
    """
    Goles esperados (λ_a, λ_b). Si neutral, promedia las dos orientaciones para no
    regalarle ventaja de localía a nadie. Si no, `a` es el local (anfitrión).
    """
    if not neutral:
        return model.expected_goals(a, b)
    # Neutral: cada equipo juega "mitad local, mitad visita" → promediamos.
    la_h, lb_a = model.expected_goals(a, b)   # a local, b visita
    lb_h, la_a = model.expected_goals(b, a)   # b local, a visita
    return (la_h + la_a) / 2.0, (lb_h + lb_a) / 2.0


def sample_score(model: PoissonModel, a: str, b: str, neutral: bool,
                 rng: np.random.Generator) -> tuple[int, int]:
    """Muestrea un marcador (goles_a, goles_b) de la distribución de Poisson."""
    la, lb = expected_goals(model, a, b, neutral)
    ga = int(rng.poisson(max(la, 1e-6)))
    gb = int(rng.poisson(max(lb, 1e-6)))
    return ga, gb


# ──────────────────────────────────────────────────────────────────────────────
# Fase de grupos
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TeamRecord:
    """Acumulador de la tabla de un equipo dentro de su grupo."""
    team: str
    points: int = 0
    gf: int = 0
    ga: int = 0
    played: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def add(self, scored: int, conceded: int) -> None:
        self.played += 1
        self.gf += scored
        self.ga += conceded
        if scored > conceded:
            self.points += 3
        elif scored == conceded:
            self.points += 1


def infer_groups(fixtures: pd.DataFrame) -> list[list[str]]:
    """Reconstruye los 12 grupos como componentes conexas del grafo de enfrentamientos."""
    from collections import defaultdict
    adj: dict[str, set[str]] = defaultdict(set)
    for r in fixtures.itertuples(index=False):
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)
    visto: set[str] = set()
    grupos: list[list[str]] = []
    for eq in adj:
        if eq in visto:
            continue
        comp, cola = set(), [eq]
        while cola:
            n = cola.pop()
            if n in comp:
                continue
            comp.add(n)
            visto.add(n)
            cola.extend(adj[n] - comp)
        grupos.append(sorted(comp))
    return grupos


def _rank_group(records: dict[str, TeamRecord],
                rng: np.random.Generator) -> list[TeamRecord]:
    """
    Ordena un grupo por criterios FIFA: puntos → diferencia de gol → goles a favor.
    Los empates que sobreviven se rompen al azar (en una simulación, eso reparte el
    azar de forma justa entre simulaciones; no introduce sesgo sistemático).
    """
    recs = list(records.values())
    # Clave de orden descendente; el ruido aleatorio final desempata lo que quede igual.
    return sorted(recs, key=lambda r: (r.points, r.gd, r.gf, rng.random()), reverse=True)


def simulate_group_rows(model: PoissonModel, rows: list[tuple[str, str, bool]],
                        rng: np.random.Generator) -> list[TeamRecord]:
    """
    Juega un grupo desde filas YA extraídas (home, away, neutral) y devuelve la tabla.
    Trabajar sobre tuplas (no DataFrame) es lo que hace viable el Monte Carlo: filtrar
    pandas dentro del loop de simulaciones es carísimo; acá las filas se pre-computan
    UNA vez y el hot loop solo hace aritmética.
    """
    teams = sorted({t for h, a, _ in rows for t in (h, a)})
    records = {t: TeamRecord(t) for t in teams}
    for home, away, neutral in rows:
        ga, gb = sample_score(model, home, away, neutral, rng)
        records[home].add(ga, gb)
        records[away].add(gb, ga)
    return _rank_group(records, rng)


def simulate_group(model: PoissonModel, fixtures: pd.DataFrame,
                   rng: np.random.Generator) -> list[TeamRecord]:
    """Versión cómoda desde DataFrame (para uso suelto y tests). Convierte a filas."""
    rows = [(r.home_team, r.away_team, bool(r.neutral))
            for r in fixtures.itertuples(index=False)]
    return simulate_group_rows(model, rows, rng)


def group_rows_by_letter(fixtures: pd.DataFrame,
                         groups: dict[str, list[str]]) -> dict[str, list[tuple[str, str, bool]]]:
    """
    Pre-computa, UNA sola vez, los partidos internos de cada grupo como listas de
    (home, away, neutral). Esto saca el filtrado de pandas del hot loop del Monte Carlo.
    """
    out: dict[str, list[tuple[str, str, bool]]] = {}
    for letra, equipos in groups.items():
        eqset = set(equipos)
        out[letra] = [(r.home_team, r.away_team, bool(r.neutral))
                      for r in fixtures.itertuples(index=False)
                      if r.home_team in eqset and r.away_team in eqset]
    return out


def simulate_group_stage(model: PoissonModel,
                         group_rows: dict[str, list[tuple[str, str, bool]]],
                         rng: np.random.Generator) -> dict[str, list[TeamRecord]]:
    """
    Simula los 12 grupos y devuelve la tabla ordenada POR LETRA de grupo (A..L).
    `group_rows` mapea letra → partidos pre-computados (home, away, neutral); usá
    group_rows_by_letter() una vez para construirlo. Trabajar por letra es lo que el
    bracket oficial necesita ("1A", "2C", "mejor 3º de {C,E,F,H,I}", etc.).
    """
    return {letra: simulate_group_rows(model, rows, rng)
            for letra, rows in group_rows.items()}


def best_eight_thirds(standings: dict[str, list[TeamRecord]],
                      rng: np.random.Generator) -> list[str]:
    """
    De los 12 terceros (uno por grupo), elige las 8 LETRAS de grupo cuyos terceros
    clasifican, por puntos → GD → GF. Devuelve letras (no equipos) porque el slotting
    del bracket se define por grupo de procedencia.
    """
    terceros = [(letra, tabla[2]) for letra, tabla in standings.items()]
    ordenados = sorted(terceros,
                       key=lambda lt: (lt[1].points, lt[1].gd, lt[1].gf, rng.random()),
                       reverse=True)
    return [letra for letra, _ in ordenados[:8]]


# ──────────────────────────────────────────────────────────────────────────────
# Carga de fixtures + modelo (helpers para el Monte Carlo)
# ──────────────────────────────────────────────────────────────────────────────
def load_world_cup_fixtures() -> pd.DataFrame:
    """Los 72 fixtures de fase de grupos del Mundial 2026 (martj42)."""
    df = InternationalResultsLoader().load(include_unplayed=True)
    wc = df[(df["date"] >= pd.Timestamp(WORLD_CUP_2026_FROM))
            & (df["tournament"].str.contains("World Cup", case=False, na=False))].copy()
    return wc.reset_index(drop=True)


def train_goal_model(since: str = "2010-01-01") -> PoissonModel:
    """Poisson de selecciones para simular goles (data moderna: las selecciones cambian)."""
    df = InternationalResultsLoader().load().dropna(subset=["home_score", "away_score"])
    return PoissonModel(max_goals=10).fit(df[df["date"] >= since])


def train_elo_goal_model(since: str = "2010-01-01"):
    """
    Modelo de goles AJUSTADO POR RIVAL (Elo) — el recomendado para el Monte Carlo.
    Corrige el sesgo del Poisson crudo, que infla a equipos que golean a rivales
    débiles de su confederación (Irán, NZ...). Misma interfaz expected_goals(a, b).
    """
    from src.models.elo_poisson import EloGoalModel
    df = InternationalResultsLoader().load().dropna(subset=["home_score", "away_score"])
    return EloGoalModel(max_goals=10).fit(df[df["date"] >= since])


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.predict.simulate
# Monte Carlo de la FASE DE GRUPOS: probabilidad de avanzar de cada selección.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    N = 5000
    rng = np.random.default_rng(42)
    model = train_goal_model()
    fixtures = load_world_cup_fixtures()

    # Para el demo standalone usamos grupos inferidos con claves genéricas (G1..G12);
    # las letras oficiales A..L viven en bracket.py y solo hacen falta para el cuadro.
    groups = {f"G{i+1}": g for i, g in enumerate(infer_groups(fixtures))}
    grows = group_rows_by_letter(fixtures, groups)   # pre-cómputo (fuera del loop)

    from collections import Counter
    avanza = Counter()      # clasifica a eliminatorias (1º, 2º o mejor 3º)
    primero = Counter()     # gana su grupo
    for _ in range(N):
        standings = simulate_group_stage(model, grows, rng)
        terceros = set(best_eight_thirds(standings, rng))
        for letra, tabla in standings.items():
            primero[tabla[0].team] += 1
            avanza[tabla[0].team] += 1
            avanza[tabla[1].team] += 1
            if letra in terceros:
                avanza[tabla[2].team] += 1

    todos = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    tabla = pd.DataFrame({
        "team": todos,
        "P(avanza)": [avanza[t] / N for t in todos],
        "P(gana grupo)": [primero[t] / N for t in todos],
    }).sort_values("P(avanza)", ascending=False).reset_index(drop=True)
    tabla["P(avanza)"] = (tabla["P(avanza)"] * 100).round(1)
    tabla["P(gana grupo)"] = (tabla["P(gana grupo)"] * 100).round(1)

    print(f"=== Fase de grupos · Monte Carlo {N} simulaciones ===\n")
    print(tabla.head(20).to_string(index=False))
