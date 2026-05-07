"""
retrieve.py  —  provided by Quant Singularity
Do not modify this file.
Place it in the same directory as rag_corpus.jsonl.
"""
import json
import numpy as np
from pathlib import Path

_CORPUS_PATH = Path(__file__).parent / "rag_corpus.jsonl"
_corpus = None
_corpus_vecs = None

_KEYS    = ["adx_14", "vix_india", "iv_skew_25d", "pcr", "dte_nearest"]
_WEIGHTS = [1.5, 2.0, 0.8, 1.0, 0.5]
_STATS   = {
    "adx_14":      {"mean": 24.0, "std": 8.0},
    "vix_india":   {"mean": 18.0, "std": 7.0},
    "iv_skew_25d": {"mean": 0.8,  "std": 2.0},
    "pcr":         {"mean": 1.05, "std": 0.25},
    "dte_nearest": {"mean": 2.0,  "std": 1.5},
}


def _load():
    global _corpus, _corpus_vecs
    if _corpus is not None:
        return
    _corpus = [json.loads(l) for l in open(_CORPUS_PATH)]
    vecs = []
    for ep in _corpus:
        ms = ep["market_state"]
        v = [(ms.get(k, 0.0) - _STATS[k]["mean"]) / _STATS[k]["std"] * w
             for k, w in zip(_KEYS, _WEIGHTS)]
        vecs.append(v)
    _corpus_vecs = np.array(vecs)


def retrieve(market_state: dict, k: int = 3) -> list:
    """
    Return the k most similar historical episodes for the given market state.

    Args:
        market_state : dict with keys matching the market state schema
        k            : number of episodes to return (default 3)

    Returns:
        List of k episode dicts, each with:
            episode_id, regime, summary, market_state,
            outcome, outcome_description
    """
    _load()
    q = np.array(
        [(market_state.get(k_, 0.0) - _STATS[k_]["mean"]) / _STATS[k_]["std"] * w
         for k_, w in zip(_KEYS, _WEIGHTS)]
    )
    dists = np.linalg.norm(_corpus_vecs - q, axis=1)
    return [_corpus[i] for i in np.argsort(dists)[:k]]
