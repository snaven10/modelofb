"""
bracket.py — Cuadro de eliminatorias OFICIAL del Mundial 2026 + Monte Carlo completo.

Sobre las primitivas de simulate.py (muestreo de marcadores con Poisson, fase de
grupos por letra) montamos el bracket OFICIAL de FIFA y simulamos el torneo entero
miles de veces para estimar, por selección, la probabilidad de llegar a cada ronda y
de salir CAMPEÓN.

FIDELIDAD (lo que es oficial y lo que aproximamos, sin maquillaje):
  - OFICIAL y verificado: las 12 letras de grupo, los 16 cruces del Round of 32
    (1E vs mejor-3º de {A,B,C,D,F}, etc.), los 5 grupos elegibles por cada slot de
    tercero, y el árbol completo R32→R16→cuartos→semis→final.
  - APROXIMADO: la tabla "Annex C" de FIFA (495 combinaciones) que asigna QUÉ tercero
    exacto va a QUÉ slot no es pública en formato usable. Asignamos cada tercero
    clasificado a un slot ELEGIBLE vía matching bipartito válido (respeta las
    restricciones reales de 5-grupos-por-slot). El cuadro resultante es válido; solo el
    desempate fino entre matchings equivalentes puede diferir del de FIFA — efecto de
    segundo orden sobre la probabilidad de campeón.

EMPATES EN ELIMINATORIAS: no hay empate. Tras los 90', si está igualado simulamos
prórroga (λ reducido a ~1/3, son 30') y, si persiste, penales (≈ moneda al aire: los
penales son casi azar, así que 50/50).
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from src.models.poisson import PoissonModel
from src.predict.simulate import (
    best_eight_thirds,
    expected_goals,
    group_rows_by_letter,
    load_world_cup_fixtures,
    sample_score,
    simulate_group_stage,
    train_elo_goal_model,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Estructura oficial (nombres de equipo EXACTOS como en martj42/international_results)
# ──────────────────────────────────────────────────────────────────────────────
GROUPS_BY_LETTER: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Bosnia and Herzegovina", "Qatar"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Colombia", "Portugal", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Round of 32 (partidos 73–88). spec = ("W",letra)=ganador, ("R",letra)=segundo,
# ("3",slot)=mejor tercero asignado a ese slot (la elegibilidad va en THIRD_ELIGIBLE).
R32: dict[int, tuple] = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("3", 74)),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("3", 77)),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("3", 79)),
    80: (("W", "L"), ("3", 80)),
    81: (("W", "D"), ("3", 81)),
    82: (("W", "G"), ("3", 82)),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("3", 85)),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("3", 87)),
    88: (("R", "D"), ("R", "G")),
}

# Grupos elegibles para cada slot de tercero (columna derecha del R32 oficial).
THIRD_ELIGIBLE: dict[int, frozenset[str]] = {
    74: frozenset("ABCDF"),
    77: frozenset("CDFGH"),
    79: frozenset("CEFHI"),
    80: frozenset("EHIJK"),
    81: frozenset("BEFIJ"),
    82: frozenset("AEHIJ"),
    85: frozenset("EFGIJ"),
    87: frozenset("DEIJL"),
}

# Llaves desde R16 hasta la final: cada partido se alimenta de los ganadores de dos
# partidos previos. (match → (feed_home, feed_away))
KO_FEEDS: dict[int, tuple[int, int]] = {
    # Round of 16
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    # Cuartos
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    # Semis
    101: (97, 98), 102: (99, 100),
    # Final
    104: (101, 102),
}

ROUND_OF = {  # para reportar hasta qué instancia llegó cada equipo
    "R32": list(R32.keys()),
    "R16": [89, 90, 91, 92, 93, 94, 95, 96],
    "QF": [97, 98, 99, 100],
    "SF": [101, 102],
    "Final": [104],
}


# ──────────────────────────────────────────────────────────────────────────────
# Asignación de mejores terceros a slots (matching bipartito válido)
# ──────────────────────────────────────────────────────────────────────────────
def assign_thirds(qualified: list[str], rng: np.random.Generator) -> dict[int, str]:
    """
    Asigna las 8 letras de grupo cuyos terceros clasificaron a los 8 slots de tercero,
    respetando la elegibilidad oficial (THIRD_ELIGIBLE). Devuelve {slot: letra}.

    Es un matching bipartito perfecto. Puede haber varios válidos; elegimos uno por
    backtracking, barajando el orden con `rng` para que distintas simulaciones exploren
    asignaciones válidas distintas (en vez de fijar siempre la misma).
    """
    slots = list(THIRD_ELIGIBLE.keys())
    rng.shuffle(slots)
    qset = set(qualified)
    asign: dict[int, str] = {}
    usadas: set[str] = set()

    def backtrack(i: int) -> bool:
        if i == len(slots):
            return True
        slot = slots[i]
        candidatas = [g for g in THIRD_ELIGIBLE[slot] if g in qset and g not in usadas]
        rng.shuffle(candidatas)
        for g in candidatas:
            asign[slot] = g
            usadas.add(g)
            if backtrack(i + 1):
                return True
            usadas.discard(g)
            del asign[slot]
        return False

    if not backtrack(0):
        # Por diseño FIFA siempre existe matching; si no (combinación rara), fallback:
        # asignar lo que se pueda y el resto en orden. No debería ocurrir.
        logger.warning("Sin matching perfecto de terceros para %s; usando fallback", qualified)
        restantes = [g for g in qualified if g not in usadas]
        for slot in slots:
            if slot not in asign and restantes:
                asign[slot] = restantes.pop()
    return asign


# ──────────────────────────────────────────────────────────────────────────────
# Partido de eliminatoria (con prórroga y penales) y resolución de specs
# ──────────────────────────────────────────────────────────────────────────────
def play_knockout(model: PoissonModel, a: str, b: str,
                  rng: np.random.Generator) -> str:
    """Juega un partido a muerte: 90' → prórroga → penales. Devuelve el ganador."""
    ga, gb = sample_score(model, a, b, neutral=True, rng=rng)
    if ga != gb:
        return a if ga > gb else b
    # Prórroga: ~1/3 de un partido (30'). Reusamos λ reducido.
    la, lb = (x / 3.0 for x in expected_goals(model, a, b, neutral=True))
    ea, eb = int(rng.poisson(max(la, 1e-6))), int(rng.poisson(max(lb, 1e-6)))
    if ea != eb:
        return a if ea > eb else b
    # Penales: casi azar.
    return a if rng.random() < 0.5 else b


def _resolve_spec(spec: tuple, standings: dict[str, list],
                  thirds: dict[int, str]) -> str:
    """Convierte un spec del R32 en un equipo concreto."""
    kind, ref = spec
    if kind == "W":
        return standings[ref][0].team
    if kind == "R":
        return standings[ref][1].team
    if kind == "3":               # ref = número de slot
        letra = thirds[ref]
        return standings[letra][2].team
    raise ValueError(f"spec desconocido: {spec}")


# ──────────────────────────────────────────────────────────────────────────────
# Una simulación completa del torneo
# ──────────────────────────────────────────────────────────────────────────────
def simulate_tournament(model: PoissonModel,
                        group_rows: dict[str, list[tuple[str, str, bool]]],
                        rng: np.random.Generator) -> dict[int, str]:
    """
    Juega el Mundial entero una vez. Devuelve {número_de_partido: ganador} para todo el
    bracket. `group_rows` se pre-computa con group_rows_by_letter() (una vez).
    """
    standings = simulate_group_stage(model, group_rows, rng)
    thirds_letters = best_eight_thirds(standings, rng)
    thirds = assign_thirds(thirds_letters, rng)

    winners: dict[int, str] = {}

    # Round of 32
    for m, (hs, as_) in R32.items():
        home = _resolve_spec(hs, standings, thirds)
        away = _resolve_spec(as_, standings, thirds)
        winners[m] = play_knockout(model, home, away, rng)

    # R16 → Final (en orden de número de partido, que respeta las dependencias)
    for m in sorted(KO_FEEDS):
        fh, fa = KO_FEEDS[m]
        winners[m] = play_knockout(model, winners[fh], winners[fa], rng)

    return winners


# ──────────────────────────────────────────────────────────────────────────────
# Monte Carlo: probabilidad por ronda y de campeón
# ──────────────────────────────────────────────────────────────────────────────
def monte_carlo(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Simula el Mundial `n` veces y devuelve, por selección, la probabilidad de alcanzar
    cada ronda (clasificar de grupos, R16, cuartos, semis, final) y de ser campeón.
    """
    rng = np.random.default_rng(seed)
    model = train_elo_goal_model()      # goles ajustados por rival (corrige el sesgo)
    fixtures = load_world_cup_fixtures()
    all_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    grows = group_rows_by_letter(fixtures, GROUPS_BY_LETTER)   # pre-cómputo, fuera del loop

    counters = {r: Counter() for r in ["R32", "R16", "QF", "SF", "Final", "Champion"]}

    for _ in range(n):
        standings = simulate_group_stage(model, grows, rng)
        thirds_letters = best_eight_thirds(standings, rng)
        thirds = assign_thirds(thirds_letters, rng)

        # Clasificados a R32 (32 equipos): 2 por grupo + 8 terceros.
        for letra, tabla in standings.items():
            counters["R32"][tabla[0].team] += 1
            counters["R32"][tabla[1].team] += 1
        for slot, letra in thirds.items():
            counters["R32"][standings[letra][2].team] += 1

        winners: dict[int, str] = {}
        for m, (hs, as_) in R32.items():
            home = _resolve_spec(hs, standings, thirds)
            away = _resolve_spec(as_, standings, thirds)
            winners[m] = play_knockout(model, home, away, rng)
        for m in sorted(KO_FEEDS):
            fh, fa = KO_FEEDS[m]
            winners[m] = play_knockout(model, winners[fh], winners[fa], rng)

        # Quién ganó en cada ronda = quién ALCANZÓ la siguiente.
        for m in ROUND_OF["R32"]:
            counters["R16"][winners[m]] += 1          # ganó R32 → está en R16
        for m in ROUND_OF["R16"]:
            counters["QF"][winners[m]] += 1
        for m in ROUND_OF["QF"]:
            counters["SF"][winners[m]] += 1
        for m in ROUND_OF["SF"]:
            counters["Final"][winners[m]] += 1
        counters["Champion"][winners[104]] += 1

    tabla = pd.DataFrame({"team": all_teams})
    for r in ["R32", "R16", "QF", "SF", "Final", "Champion"]:
        tabla[r] = [counters[r][t] / n for t in all_teams]
    return tabla.sort_values("Champion", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  python -m src.predict.bracket
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    pd.set_option("display.width", 140)

    N = 10000
    print(f"Simulando el Mundial 2026 · {N} veces (Monte Carlo)...\n")
    tabla = monte_carlo(n=N)

    pct = tabla.copy()
    for c in ["R32", "R16", "QF", "SF", "Final", "Champion"]:
        pct[c] = (pct[c] * 100).round(1)
    print("=== Probabilidad por ronda (%) — top 16 candidatos al título ===\n")
    print(pct.head(16).to_string(index=False))
    campeon = tabla.iloc[0]
    print(f"\n🏆 Favorito al título: {campeon['team']} ({campeon['Champion']*100:.1f}%)")
