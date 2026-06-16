"""
elo.py — Modelo Elo para selecciones de fútbol.

Este es el "hola mundo" de la predicción deportiva. Un número por selección (su
rating) que sube o baja según los resultados. Acá implementamos EXACTAMENTE los
conceptos que discutimos antes de tocar código:

  1. PROBABILIDAD ESPERADA (curva logística, escala 400):
         E_local = 1 / (1 + 10^(-(dif)/400))
     donde `dif` es la diferencia de ratings (con la ventaja de localía sumada al
     local, SALVO que el partido sea en cancha neutral).

  2. ACTUALIZACIÓN (la magia — el sistema aprende solo):
         R_nuevo = R_viejo + K · (S − E)
     S = resultado real (1 gana / 0.5 empata / 0 pierde). (S − E) es la SORPRESA:
     batacazos mueven mucho, resultados esperados mueven poco. Es suma cero: lo que
     gana un equipo lo pierde el otro.

  3. K MODULADO POR IMPORTANCIA:
     Un amistoso pesa menos que un partido competitivo. K_efectivo = K · factor.

  4. EL EMPATE (el talón de Aquiles del Elo, heredado del ajedrez):
     El Elo da un "expected score" que mezcla victoria y empate. Para sacar las TRES
     probabilidades (local / empate / visitante) modelamos el empate aparte: es más
     probable cuanto más PAREJO está el partido. Ver predict_proba().

LIMITACIÓN HONESTA: el Elo predice quién gana, NO marcadores ni goles. Para eso
viene después Poisson/Dixon-Coles. Esto es el cimiento, no el techo.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EloModel:
    """Ratings Elo para selecciones, con ventaja de localía e importancia de partido."""

    def __init__(
        self,
        k_base: float = 20.0,
        home_advantage: float = 100.0,
        friendly_weight: float = 0.5,
        base_rating: float = 1500.0,
        scale: float = 400.0,
        draw_max: float = 0.28,
    ) -> None:
        """
        Parámetros (todos CALIBRABLES después, en la Fase 4):
          k_base          cuánto se mueve un rating por partido (reactividad vs estabilidad).
          home_advantage  bonus de puntos al local cuando NO es cancha neutral (~100).
          friendly_weight factor de K para amistosos (<1: pesan menos que los competitivos).
          base_rating     rating inicial de toda selección nueva (convención: 1500).
          scale           la escala de la curva logística (convención: 400).
          draw_max        probabilidad de empate cuando el partido está MÁXIMO de parejo.
        """
        self.k_base = k_base
        self.home_advantage = home_advantage
        self.friendly_weight = friendly_weight
        self.base_rating = base_rating
        self.scale = scale
        self.draw_max = draw_max

        # rating actual de cada selección. Las que no existen aún arrancan en base_rating.
        self.ratings: dict[str, float] = {}

        # Coeficientes de la curva de empate p_draw(d) = c0 + c1·d + c2·d², con
        # d=|2E−1| (el desbalance). Se calibran en fit() desde los datos reales. Mientras
        # sea None, predict_proba usa la heurística vieja draw_max·(1−d).
        self.draw_coef: tuple[float, float, float] | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Acceso a ratings
    # ──────────────────────────────────────────────────────────────────────
    def get_rating(self, team: str) -> float:
        """Rating actual del equipo; si nunca jugó, devuelve el base (1500)."""
        return self.ratings.get(team, self.base_rating)

    # ──────────────────────────────────────────────────────────────────────
    # 1. Probabilidad esperada (curva logística)
    # ──────────────────────────────────────────────────────────────────────
    def expected_score(self, home: str, away: str, neutral: bool = False) -> float:
        """
        Expected score del LOCAL: número entre 0 y 1 que mezcla P(gana)+0.5·P(empata).
        Es la fórmula logística con la ventaja de localía aplicada salvo en neutral.
        """
        # La ventaja de localía SOLO cuenta si NO es cancha neutral (clave para el Mundial).
        bonus = 0.0 if neutral else self.home_advantage
        diff = (self.get_rating(home) + bonus) - self.get_rating(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    # ──────────────────────────────────────────────────────────────────────
    # 4. Probabilidades 1X2 (modelando el empate aparte)
    # ──────────────────────────────────────────────────────────────────────
    def predict_proba(
        self, home: str, away: str, neutral: bool = False
    ) -> tuple[float, float, float]:
        """
        Devuelve (p_local, p_empate, p_visita), que suman 1.

        CÓMO MODELAMOS EL EMPATE: el Elo nos da E = expected score del local, que por
        definición es  E = p_local + 0.5·p_empate. Eso es UNA ecuación con dos
        incógnitas; nos falta otra. La aportamos con una heurística sensata:

            p_empate = draw_max · (1 − |2E − 1|)

        Cuando el partido está parejo (E≈0.5) el término |2E−1|≈0 → empate máximo.
        Cuando hay un claro favorito (E→0 o E→1) → el empate tiende a 0. Es justo lo
        que pasa en el fútbol real: los partidos parejos empatan más.

        Con esa segunda ecuación, despejamos:
            p_local = E − 0.5·p_empate
            p_visita = (1 − E) − 0.5·p_empate
        (Se verifica fácil que las tres suman 1.)
        """
        e = self.expected_score(home, away, neutral)
        p_draw = self._draw_prob(e)
        p_home = e - 0.5 * p_draw
        p_away = (1.0 - e) - 0.5 * p_draw

        # Clamp defensivo: con parámetros sanos no debería activarse, pero si alguien
        # mete un draw_max absurdo evitamos probabilidades negativas.
        p_home = max(p_home, 0.0)
        p_away = max(p_away, 0.0)
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def _draw_prob(self, e: float) -> float:
        """
        Probabilidad de empate dada E (expected score). Si la curva está calibrada
        (draw_coef), usa la cuadrática ajustada a los datos; si no, la heurística vieja.

        Los datos reales muestran que la tasa de empate es PLANA (~25%) salvo en
        partidos muy desbalanceados, donde baja a ~12% — no el decaimiento lineal
        brusco de la heurística. La cuadrática captura esa meseta + caída.
        """
        d = abs(2.0 * e - 1.0)
        if self.draw_coef is not None:
            c0, c1, c2 = self.draw_coef
            return float(min(max(c0 + c1 * d + c2 * d * d, 0.02), 0.45))
        return self.draw_max * (1.0 - d)

    def _calibrate_draw(self, df: pd.DataFrame) -> None:
        """
        Ajusta p_draw(d) = c0 + c1·d + c2·d² por mínimos cuadrados sobre los partidos
        históricos: d=|2E−1| (con los ratings ya entrenados como proxy de fuerza) e
        y=1 si fue empate. Es una RELACIÓN agregada (la forma de la curva), no predicción
        de partidos individuales, así que usar ratings finales como proxy es admisible.
        """
        d_vals, y = [], []
        for row in df.itertuples(index=False):
            if pd.isna(row.home_score) or pd.isna(row.away_score):
                continue
            h, a = row.home_team, row.away_team
            if h not in self.ratings or a not in self.ratings:
                continue
            e = self.expected_score(h, a, bool(getattr(row, "neutral", False)))
            d_vals.append(abs(2.0 * e - 1.0))
            y.append(1.0 if row.home_score == row.away_score else 0.0)

        if len(d_vals) < 200:          # con pocos datos no calibramos; queda la heurística
            return
        d_arr = np.array(d_vals)
        X = np.column_stack([np.ones_like(d_arr), d_arr, d_arr ** 2])
        coef, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
        self.draw_coef = (float(coef[0]), float(coef[1]), float(coef[2]))
        logger.info("Curva de empate calibrada: p_draw(d)=%.3f%+.3f·d%+.3f·d²",
                    *self.draw_coef)

    # ──────────────────────────────────────────────────────────────────────
    # 2 + 3. Actualización tras un partido
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _actual_score(home_score: int, away_score: int) -> float:
        """S del LOCAL: 1 si gana, 0.5 si empata, 0 si pierde."""
        if home_score > away_score:
            return 1.0
        if home_score < away_score:
            return 0.0
        return 0.5

    def update_one(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        neutral: bool = False,
        is_competitive: bool = True,
    ) -> None:
        """
        Procesa UN partido: calcula la sorpresa y mueve ambos ratings.

        El orden importa: PRIMERO calculamos lo esperado con los ratings actuales,
        DESPUÉS movemos. Y es suma cero: el delta que gana el local es exactamente el
        que pierde el visitante (por eso restamos el mismo número a uno y otro).
        """
        e_home = self.expected_score(home, away, neutral)
        s_home = self._actual_score(home_score, away_score)

        # K modulado por importancia: amistoso pesa menos.
        k = self.k_base * (1.0 if is_competitive else self.friendly_weight)

        delta = k * (s_home - e_home)
        self.ratings[home] = self.get_rating(home) + delta
        self.ratings[away] = self.get_rating(away) - delta  # suma cero

    # ──────────────────────────────────────────────────────────────────────
    # Entrenamiento sobre un dataset completo
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "EloModel":
        """
        Procesa TODOS los partidos en ORDEN CRONOLÓGICO, actualizando ratings.

        El orden cronológico es OBLIGATORIO: el Elo aprende del pasado para predecir
        el futuro. Si procesaras desordenado, estarías usando partidos futuros para
        "explicar" partidos pasados = ver el futuro = trampa. El DataFrame de nuestros
        loaders ya viene ordenado por fecha, pero lo reordenamos por las dudas.

        Espera las columnas de nuestro esquema común:
            home_team, away_team, home_score, away_score, neutral, is_competitive.
        """
        ordered = df.sort_values("date") if "date" in df.columns else df
        procesados = 0
        for row in ordered.itertuples(index=False):
            # Saltamos partidos sin marcador (fixtures futuros): no se puede aprender de ellos.
            if pd.isna(row.home_score) or pd.isna(row.away_score):
                continue
            self.update_one(
                home=row.home_team,
                away=row.away_team,
                home_score=int(row.home_score),
                away_score=int(row.away_score),
                neutral=bool(getattr(row, "neutral", False)),
                is_competitive=bool(getattr(row, "is_competitive", True)),
            )
            procesados += 1

        logger.info("Elo entrenado con %d partidos · %d selecciones con rating",
                    procesados, len(self.ratings))

        # Con los ratings ya entrenados, calibramos la curva de empate desde los datos.
        self._calibrate_draw(ordered)
        return self

    def ranking(self, top: int | None = None) -> pd.DataFrame:
        """Tabla de selecciones ordenada por rating (de mayor a menor)."""
        tabla = (
            pd.DataFrame({"team": list(self.ratings.keys()),
                          "rating": list(self.ratings.values())})
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )
        tabla["rating"] = tabla["rating"].round(1)
        return tabla.head(top) if top else tabla


# ──────────────────────────────────────────────────────────────────────────────
# Demo a mano:  python -m src.models.elo
# Entrena con los 49k internacionales reales y muestra el top-20 mundial.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_loaders.international_loader import InternationalResultsLoader

    df = InternationalResultsLoader().load()
    elo = EloModel().fit(df)

    print("\n=== TOP 20 selecciones por Elo ===")
    print(elo.ranking(top=20).to_string(index=False))

    # Sanity check de una predicción concreta en cancha neutral (como el Mundial).
    print("\n=== Ejemplo: Brasil vs Argentina (neutral) ===")
    ph, pd_, pa = elo.predict_proba("Brazil", "Argentina", neutral=True)
    print(f"Brasil gana: {ph:.1%}  |  Empate: {pd_:.1%}  |  Argentina gana: {pa:.1%}")
