"""Parameter tuning by coordinate descent on Brier score, with a held-out validation split.
Tune on early weeks, confirm on later weeks -- if holdout doesn't improve, the change is overfit.

Usage: python tune.py --season 2025 --train 4-11 --test 12-18
"""
import argparse

import numpy as np

import backtest as B
import config as C
from game import Model

SEARCH = {
    "TARGET_SHARE_CONC": [15, 20, 30, 45, 70],
    "CARRY_SHARE_CONC": [12, 18, 25, 40],
    "POINTS_PER_TD": [7.0, 7.3, 7.7, 8.1],
    "FUNNEL_BETA": [0.0, 0.01, 0.02, 0.035],
    "SCRIPT_PASS_BETA": [0.03, 0.05, 0.075, 0.10],
    "YPR_GAMMA_SHAPE": [1.0, 1.35, 1.8],
    "RUSH_SD_PER_CARRY": [4.5, 5.8, 7.0],
    "EFFICIENCY_POINTS_ELASTICITY": [0.2, 0.35, 0.5],
}


def objective(prep, params, n):
    sc = B.score(B.collect_prepared(prep, n=n, params=params))
    return float(sc["brier"].mean()), sc


def coordinate_descent(prep, n=2000, passes=2, verbose=True):
    best = {k: getattr(C, k) for k in SEARCH}
    best_score, _ = objective(prep, best, n)
    if verbose:
        print(f"start {best_score:.5f}", flush=True)
    for _ in range(passes):
        for k, vals in SEARCH.items():
            for v in vals:
                if v == best[k]:
                    continue
                trial = {**best, k: v}
                s, _ = objective(prep, trial, n)
                if s < best_score - 1e-5:
                    best, best_score = trial, s
                    if verbose:
                        print(f"  {k}={v} -> {s:.5f}", flush=True)
    return best, best_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--train", default="4-11")
    ap.add_argument("--test", default="12-18")
    args = ap.parse_args()
    rng = lambda s: list(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1))
    m = Model(args.season)
    prep_tr = B.prepare(m, rng(args.train))
    prep_te = B.prepare(m, rng(args.test))
    default = {k: getattr(C, k) for k in SEARCH}
    best, s = coordinate_descent(prep_tr)
    d_te, sc_d = objective(prep_te, default, 3000)
    b_te, sc_b = objective(prep_te, best, 3000)
    print("\nbest params:", best)
    print(f"holdout mean Brier  default {d_te:.5f}  ->  tuned {b_te:.5f}")
    print(sc_b.to_string(index=False))
