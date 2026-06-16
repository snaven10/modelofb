"""
odds_api_loader.py — Cargador de The Odds API (odds de selecciones para validar).

POR QUÉ Y EL PATRÓN DE CACHÉ (igual que StatsBomb / API-Football):
    The Odds API regala 500 créditos/mes. Cada llamada a /odds cuesta créditos, así
    que la CACHÉ EN DISCO es obligatoria: la primera vez bajamos y guardamos el JSON;
    las siguientes leemos del disco (0 créditos). El endpoint /sports NO gasta cuota.

QUÉ EXPONE:
    - sports(): lista de competiciones disponibles (gratis).
    - world_cup_winner_market(): probabilidad IMPLÍCITA del mercado por selección,
      derivada de las odds outright del campeón (1/odd normalizada = quita el overround).
    - world_cup_match_market(): odds 1X2 de los partidos del Mundial → prob de mercado.

SEGURIDAD: la API key se lee de ODDS_API_KEY (.env) y NUNCA se imprime ni se guarda en
el nombre del archivo de caché (el cache key es un hash de la URL sin la key).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/raw/odds_api")
BASE = "https://api.the-odds-api.com/v4"


def _implied_probs(odds_by_team: dict[str, list[float]]) -> dict[str, float]:
    """Odds decimales (promedio entre casas) → prob implícita normalizada (sin overround)."""
    avg = {t: sum(v) / len(v) for t, v in odds_by_team.items() if v}
    raw = {t: 1.0 / o for t, o in avg.items() if o > 0}
    s = sum(raw.values())
    return {t: p / s for t, p in raw.items()} if s > 0 else {}


class OddsApiLoader:
    """Acceso a The Odds API con caché en disco. Lee la key de ODDS_API_KEY (.env)."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR,
                 use_cache: bool = True, env_path: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        load_dotenv(env_path or str(Path(__file__).resolve().parents[2] / ".env"))

    def _key(self) -> str:
        k = os.environ.get("ODDS_API_KEY", "").strip()
        if not k:
            raise RuntimeError("Falta ODDS_API_KEY en el entorno (.env).")
        return k

    def _get(self, path: str, params: dict, cost_note: str = "") -> object:
        """GET con caché. El nombre de caché NO incluye la key (hash de path+params)."""
        # cache key sobre path+params SIN la apiKey (no exponer el secreto en disco)
        sig = hashlib.sha1(f"{path}?{sorted(params.items())}".encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"{path.strip('/').replace('/','_')}_{sig}.json"
        if self.use_cache and cache_file.exists():
            logger.info("Odds cacheadas: %s (0 créditos)", cache_file.name)
            return json.loads(cache_file.read_text())

        query = "&".join(f"{k}={v}" for k, v in {**params, "apiKey": self._key()}.items())
        url = f"{BASE}/{path}?{query}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as r:
                data = json.load(r)
                rem = r.headers.get("x-requests-remaining")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Odds API HTTP {e.code}: {e.read().decode()[:200]}") from e
        logger.info("Odds API · %s · créditos restantes: %s%s", path, rem,
                    f" ({cost_note})" if cost_note else "")
        cache_file.write_text(json.dumps(data))
        return data

    # ── Endpoints ────────────────────────────────────────────────────────────
    def sports(self) -> list[dict]:
        """Competiciones disponibles (no gasta cuota)."""
        return self._get("sports", {}, cost_note="gratis")

    def world_cup_winner_market(self) -> dict[str, float]:
        """Prob implícita de campeón por selección (odds outright, promedio de casas)."""
        data = self._get("sports/soccer_fifa_world_cup_winner/odds",
                         {"regions": "eu", "markets": "outrights", "oddsFormat": "decimal"},
                         cost_note="~1 crédito")
        odds: dict[str, list[float]] = defaultdict(list)
        for ev in data:
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk["key"] == "outrights":
                        for o in mk["outcomes"]:
                            odds[o["name"]].append(o["price"])
        return _implied_probs(odds)

    def world_cup_match_market(self) -> pd.DataFrame:
        """Odds 1X2 de los partidos del Mundial → prob de mercado (H/D/A) por partido."""
        data = self._get("sports/soccer_fifa_world_cup/odds",
                         {"regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
                         cost_note="~1 crédito")
        filas = []
        for ev in data:
            home, away = ev["home_team"], ev["away_team"]
            o: dict[str, list[float]] = defaultdict(list)
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk["key"] == "h2h":
                        for out in mk["outcomes"]:
                            o[out["name"]].append(out["price"])
            imp = _implied_probs(o)
            if home in imp and away in imp:
                filas.append({
                    "home_team": home, "away_team": away,
                    "commence": ev.get("commence_time"),
                    "p_market_h": imp[home], "p_market_a": imp[away],
                    "p_market_d": imp.get("Draw", 0.0),
                })
        return pd.DataFrame(filas)
