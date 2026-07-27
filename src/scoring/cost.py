"""Cost model e threshold optimization per il flag di rischio NAP.

Il layer di scoring produce un `total_score` continuo. Qui quel punteggio
diventa una decisione binaria (flag / no-flag) scegliendo la soglia che
minimizza un cost model ASIMMETRICO, invece di usare soglie fisse a mano.

Perche' asimmetrico: mancare un caso di abuso (false negative) e'
clinicamente molto piu' grave che generare un falso allarme (false
positive). Il rapporto c_fn / c_fp e' una scelta di dominio, da motivare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

C_FN_DEFAULT = 10.0   # costo di un abuso NON intercettato
C_FP_DEFAULT = 1.0    # costo di un falso allarme


def binarize_labels(tipologia, positive=("sospetto", "sicuro")):
    """Label multi-classe (negativo/sospetto/sicuro) -> y_true 0/1."""
    s = pd.Series(tipologia).astype(str).str.lower()
    wanted = [p.lower() for p in positive]
    return s.isin(wanted).astype(int).to_numpy()


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def cost_model(y_true, y_pred, c_fn=C_FN_DEFAULT, c_fp=C_FP_DEFAULT):
    c = confusion_counts(y_true, y_pred)
    return c_fn * c["fn"] + c_fp * c["fp"]


def optimize_threshold(scores, y_true, c_fn=C_FN_DEFAULT, c_fp=C_FP_DEFAULT,
                       grid=None, n_grid=200):
    """Linear search della soglia che minimizza il cost model.

    Ritorna dict: best_threshold, best_cost, grid, costs, confusion.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true)

    if grid is None:
        lo, hi = float(scores.min()), float(scores.max())
        pad = (hi - lo) * 0.01 if hi > lo else 1e-3
        grid = np.linspace(lo - pad, hi + pad, n_grid)

    costs = np.array([
        cost_model(y_true, (scores >= t).astype(int), c_fn, c_fp)
        for t in grid
    ])
    best_idx = int(np.argmin(costs))
    best_t = float(grid[best_idx])
    y_best = (scores >= best_t).astype(int)

    return {
        "best_threshold": best_t,
        "best_cost": float(costs[best_idx]),
        "grid": grid,
        "costs": costs,
        "confusion": confusion_counts(y_true, y_best),
    }
