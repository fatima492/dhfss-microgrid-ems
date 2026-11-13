"""Scenario pipeline: tuning on training years, simulation on test years, block criteria and scores.

Revision: the battery rules have two free settings, the off-peak grid-charging target
(off, 30, 45, 60, 75, 90 %) and the discharge mode (on-peak plus above-limit deficits, or any deficit).
They are tuned on the training data under two objectives from the same set of simulations:
* cost objective (A2, A4): minimum operating cost including the unserved-energy penalty;
* multi-criteria objective (A5, A6): maximum weighted-sum score W = sum_j w_j s_j of the decision
  criteria on the training data (baseline criteria, criterion weights and temporal weights). With
  reference-point normalization W depends only on the policy's own outcomes, and by Proposition 5
  it orders policies exactly as MABAC does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ems_data as E
import ems_model as M
import mcdm as D

PERIODS = E.PERIODS
ALTS = M.POLICIES
OMEGA_DURATION = E.PERIOD_HOURS / E.PERIOD_HOURS.sum()          # (8, 10, 6) / 24
W_EQUAL = np.full(len(M.CRITERIA), 1.0 / len(M.CRITERIA))
GRID_TARGETS = ["off", 0.30, 0.45, 0.60, 0.75, 0.90]
DIS_MODES = ["onpeak", "any"]
SETTINGS = [(m, t) for m in DIS_MODES for t in GRID_TARGETS]
TUNED = {"A2": ("cost", "A2"), "A4": ("cost", "A4"), "A5": ("mc", "A2"), "A6": ("mc", "A4")}

_INPUT_CACHE = {}


def inputs(load_model="stochastic", pv_scale=1.0, wind_scale=1.0, prices=None, site="zurich"):
    key = (site, load_model, pv_scale, wind_scale)
    if key not in _INPUT_CACHE:
        _INPUT_CACHE[key] = (E.build_inputs_sydney(pv_scale) if site == "sydney"
                             else E.build_inputs(pv_scale, wind_scale, load_model))
    df = _INPUT_CACHE[key]
    if prices is not None:
        df = df.copy(); df["price"] = df["period"].map(prices).astype(float)
    return df


def with_setting(p, setting):
    mode, target = setting
    q = dict(p); q["discharge_mode"] = mode
    if target == "off":
        q["grid_charging"] = False
    else:
        q["grid_charging"] = True; q["grid_charge_target"] = float(target)
    return q


def setting_label(setting):
    mode, target = setting
    t = "off" if target == "off" else f"{100*float(target):.0f}%"
    return f"{'on-peak' if mode == 'onpeak' else 'any deficit'} / {t}"


def mc_objective(blocks, criteria=None, omega=None, w=None):
    criteria = criteria or M.CRITERIA
    omega = OMEGA_DURATION if omega is None else omega
    w = np.full(len(criteria), 1.0 / len(criteria)) if w is None else w
    X, _ = D.block_tensor({"x": blocks}, ["x"], criteria, PERIODS)
    return float(D.scores_quantile(X, omega)[0] @ w)


def tune(df_train, p, criteria=None, w=None):
    """Settings for A2/A4 (cost objective) and A5/A6 (multi-criteria objective), from one set of runs."""
    rows = []
    for hw in ("A2", "A4"):
        for st in SETTINGS:
            q = with_setting(p, st)
            sim = M.simulate(df_train, hw, q)
            b = M.block_criteria(sim, df_train, q)
            rows.append({"hardware": hw, "mode": st[0], "target": st[1], "train_cost_usd": sim["cost"].sum(),
                         "train_W": mc_objective(b, criteria, None, w)})
    table = pd.DataFrame(rows)
    chosen = {}
    for a, (obj, hw) in TUNED.items():
        sub = table[table.hardware == hw]
        r = sub.loc[sub.train_cost_usd.idxmin()] if obj == "cost" else sub.loc[sub.train_W.idxmax()]
        chosen[a] = (r["mode"], r["target"])
    return chosen, table


def split(df_all, site):
    if site == "sydney":
        return E.sydney_split(df_all)
    return df_all[df_all.index.year.isin(E.TRAIN_YEARS)], df_all[df_all.index.year.isin(E.TEST_YEARS)]


def run(params=None, load_model="stochastic", pv_scale=1.0, wind_scale=1.0, prices=None,
        with_lp=False, with_forecast=False, site="zurich", criteria=None, w=None):
    p = dict(M.BASE); p.update(params or {})
    df_all = inputs(load_model, pv_scale, wind_scale, prices, site)
    df_tr, df_te = split(df_all, site)
    targets, tune_table = tune(df_tr, p, criteria, w)
    sims = {}
    for a in ALTS:
        hw = TUNED[a][1] if a in TUNED else a
        pa = with_setting(p, targets[a]) if a in targets else p
        sims[a] = M.simulate(df_te, hw, pa); sims[a]["policy"] = a
    if with_lp:
        sims["L2"] = M.lp_benchmark(df_te, p, True, False, "L2")
        sims["L3"] = M.lp_benchmark(df_te, p, False, True, "L3")
        sims["L4"] = M.lp_benchmark(df_te, p, True, True, "L4")
    if with_forecast:
        sims["L2F"] = M.forecast_dispatch(df_te, p, False, "L2F")
        sims["L4F"] = M.forecast_dispatch(df_te, p, True, "L4F")
    blocks = {k: M.block_criteria(s, df_te, p) for k, s in sims.items()}
    return {"p": p, "df": df_te, "df_all": df_all, "sims": sims, "blocks": blocks,
            "targets": targets, "tune_table": tune_table}


def tensor(blocks, alts=ALTS, criteria=M.CRITERIA):
    return D.block_tensor(blocks, alts, criteria, PERIODS)


def base_Q(blocks, omega=OMEGA_DURATION, w=None, alts=ALTS, criteria=M.CRITERIA):
    w = np.full(len(criteria), 1.0 / len(criteria)) if w is None else w
    X, days = tensor(blocks, alts, criteria)
    S = D.scores_quantile(X, omega)
    return D.mabac(S, w)[0], S, X, days
