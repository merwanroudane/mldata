"""Cached synthetic data generators used across the lessons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------
# 1. Simple linear relationship  (used by the live training loop)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600, max_entries=64)
def linear_sample(n: int = 60, slope: float = 2.4, intercept: float = 5.0,
                  noise: float = 1.8, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 10.0, n)
    y = intercept + slope * x + rng.normal(0.0, noise, n)
    return pd.DataFrame({"x": x, "y": y}).sort_values("x").reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Curved relationship  (used by over/under-fitting demos)
# ----------------------------------------------------------------------
def _true_curve(x: np.ndarray) -> np.ndarray:
    return np.sin(1.35 * x) + 0.22 * x


@st.cache_data(ttl=3600, max_entries=64)
def wave_sample(n: int = 40, noise: float = 0.28, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 6.4, n))
    y = _true_curve(x) + rng.normal(0.0, noise, n)
    return pd.DataFrame({"x": x, "y": y})


def true_curve(x: np.ndarray) -> np.ndarray:
    return _true_curve(x)


# ----------------------------------------------------------------------
# 3. Binary classification scores  (used by the metrics page)
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600, max_entries=64)
def score_sample(n: int = 400, positive_rate: float = 0.35,
                 separation: float = 1.6, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < positive_rate).astype(int)
    centre = np.where(y == 1, separation, -separation)
    logit = centre + rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame({"y": y, "p": p})


# ----------------------------------------------------------------------
# 4. A tiny labelled table used to *show* what a dataset looks like
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
def toy_table(n: int = 12, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.integers(60, 220, n)
    rooms = np.clip((area // 45) + rng.integers(0, 2, n), 1, 6)
    age = rng.integers(0, 40, n)
    price = (
        18.0
        + 0.62 * area
        + 9.0 * rooms
        - 0.75 * age
        + rng.normal(0, 12, n)
    ).round(1)
    return pd.DataFrame(
        {
            "area_m2": area,
            "rooms": rooms,
            "age_years": age,
            "price_k": price,
        }
    )
