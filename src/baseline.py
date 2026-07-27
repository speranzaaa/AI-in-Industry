"""Baseline semplici (rasoio di Occam) contro cui misurare KDE + LLM.

Ogni baseline produce un PUNTEGGIO continuo indicizzato su ID_PAZIENTE, cosi'
passa per lo stesso optimize_threshold / cost_model dei layer complessi:
confronto a parita' di metodo di valutazione.
"""
from __future__ import annotations

import pandas as pd


def score_visit_frequency(test_kde):
    """Baseline comportamentale: n. di accessi in 90 giorni. Zero training."""
    df = test_kde[["ID_PAZIENTE", "num_visits_90d_t"]].copy()
    return df.set_index("ID_PAZIENTE")["num_visits_90d_t"].astype(float)


def scores_from_results(test_results):
    """Estrae i punteggi dei layer da test_results.parquet per l'ablation.

    Ritorna DataFrame (index=ID_PAZIENTE) con: solo_kde, solo_llm, combined.
    Nessuna GPU: riusa i punteggi gia' calcolati.
    """
    r = test_results.set_index("ID_PAZIENTE")
    out = pd.DataFrame(index=r.index)
    out["solo_kde"] = r["kde_score"].astype(float)
    out["solo_llm"] = r["llm_score"].astype(float)
    out["combined"] = out["solo_kde"] + out["solo_llm"]
    return out
