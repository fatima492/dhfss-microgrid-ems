"""Scenario pipeline: tuning on training years, simulation on test years, block criteria and scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

import ems_data as E
import ems_model as M
import mcdm as D

PERIODS = E.PERIODS
ALTS = M.POLICIES
OMEGA_DURATION = E.PERIOD_HOURS / E.PERIOD_HOURS.sum()          # (8, 10, 6) / 24
W_EQUAL = np.full(5, 0.2)
GRID_TARGETS = ["off", 0.30, 0.45, 0.60, 0.75, 0.90]

_INPUT_CACHE = {}


def inputs(load_model="stochastic", pv_scale=1.0, wind_scale=1.0, prices=None):
    key = (load_model, pv_scale, wind_scale)
    if key not in _INPUT_CACHE:
        _INPUT_CACHE[key] = E.build_inputs(pv_scale, wind_scale, load_model)
    df = _INPUT_CACHE[key]
    if prices is not None:
        df = df.copy(); df["price"] = df["period"].map(prices).astype(float)
    return df


def _with_target(p, target):
    q = dict(p)
    if target == "off":
        q["grid_charging"] = False
    else:
        q["grid_charging"] = True; q["grid_charge_target"] = float(target)
    return q


def tune(df_train, p):
    """Grid-charging target for A2 and A4 minimising training-period operating cost (incl. penalty)."""
    chosen, table = {}, []
    for a in ("A2", "A4"):
        best = None
        for tg in GRID_TARGETS:
            c = M.simulate(df_train, a, _with_target(p, tg))["cost"].sum()
            table.append({"policy": a, "target": tg, "train_cost_usd": c})
            if best is None or c < best[1] - 1e-9:
                best = (tg, c)
        chosen[a] = best[0]
    return chosen, pd.DataFrame(table)


def run(params=None, load_model="stochastic", pv_scale=1.0, wind_scale=1.0, prices=None,
        with_lp=False, retune=True, fixed_targets=None):
    p = dict(M.BASE); p.update(params or {})
    df_all = inputs(load_model, pv_scale, wind_scale, prices)
    df_tr = df_all[df_all.index.year.isin(E.TRAIN_YEARS)]
    df_te = df_all[df_all.index.year.isin(E.TEST_YEARS)]
    if retune:
        targets, tune_table = tune(df_tr, p)
    else:
        targets, tune_table = (fixed_targets or {"A2": p["grid_charge_target"], "A4": p["grid_charge_target"]}), None
    sims = {}
    for a in ALTS:
        pa = _with_target(p, targets[a]) if a in targets else p
        sims[a] = M.simulate(df_te, a, pa)
    if with_lp:
        sims["L2"] = M.lp_benchmark(df_te, p, True, False, "L2")
        sims["L3"] = M.lp_benchmark(df_te, p, False, True, "L3")
        sims["L4"] = M.lp_benchmark(df_te, p, True, True, "L4")
    blocks = {k: M.block_criteria(s, df_te, p) for k, s in sims.items()}
    return {"p": p, "df": df_te, "df_all": df_all, "sims": sims, "blocks": blocks,
            "targets": targets, "tune_table": tune_table}


def tensor(blocks, alts=ALTS, criteria=M.CRITERIA):
    return D.block_tensor(blocks, alts, criteria, PERIODS)


def base_Q(blocks, omega=OMEGA_DURATION, w=W_EQUAL, alts=ALTS, criteria=M.CRITERIA):
    X, days = tensor(blocks, alts, criteria)
    S = D.scores_quantile(X, omega)
    return D.mabac(S, w)[0], S, X, days
