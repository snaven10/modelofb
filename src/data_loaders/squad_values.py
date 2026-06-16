"""
squad_values.py — Valor de mercado total de cada selección del Mundial 2026.

Fuente: Transfermarkt vía PlanetFootball (snapshot junio 2026). Es la ÚNICA métrica de
"fuerza de plantel" que cubre las 48 selecciones (incluidas las chicas: Curaçao, Cape
Verde, Haití), porque el valor TOTAL por equipo suma a todos los convocados.

OJO — es un SNAPSHOT estático (no hay histórico de valores), así que sirve para ajustar
la predicción del Mundial 2026, NO para un walk-forward histórico. Y el valor de mercado
sub-valora talento de selecciones menores cuyos jugadores no juegan en ligas vistas:
sesgo conocido, no eliminable. Valores en MILLONES de euros.
"""

from __future__ import annotations

# Nombres normalizados al esquema de martj42 (Curaçao con cedilla, Bosnia "and", etc.)
SQUAD_VALUE_M_EUR: dict[str, float] = {
    "France": 1520.0, "England": 1360.0, "Spain": 1220.0, "Portugal": 1010.0,
    "Germany": 947.0, "Brazil": 928.2, "Argentina": 807.5, "Netherlands": 754.2,
    "Norway": 589.9, "Belgium": 547.5, "Ivory Coast": 522.1, "Senegal": 478.1,
    "Turkey": 473.7, "Morocco": 447.7, "Sweden": 406.08, "Croatia": 387.3,
    "United States": 385.6, "Ecuador": 368.7, "Uruguay": 359.3, "Switzerland": 332.5,
    "Colombia": 302.35, "Japan": 270.85, "Algeria": 256.9, "Austria": 245.2,
    "Ghana": 234.5, "Canada": 198.65, "Mexico": 191.85, "Czech Republic": 188.18,
    "Scotland": 170.25, "Paraguay": 153.65, "Bosnia and Herzegovina": 146.4,
    "DR Congo": 143.9, "South Korea": 139.05, "Egypt": 116.48, "Uzbekistan": 85.33,
    "Australia": 77.45, "Tunisia": 69.95, "Haiti": 55.9, "Cape Verde": 49.25,
    "South Africa": 49.25, "Saudi Arabia": 40.68, "Panama": 34.55, "New Zealand": 34.45,
    "Iran": 32.05, "Curaçao": 25.78, "Iraq": 21.2, "Jordan": 20.3, "Qatar": 19.93,
}


def squad_values() -> dict[str, float]:
    """Valor de plantel (M€) por selección del Mundial 2026."""
    return dict(SQUAD_VALUE_M_EUR)
