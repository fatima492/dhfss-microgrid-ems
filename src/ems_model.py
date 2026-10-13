"""Rule-based EMS policies, perfect-foresight LP ceilings, storage provenance and block criteria.

Energy balance at every 15-min step (kW):
    renewable:  P_RE = P_RE->load + P_ch,RE + P_curt
    demand:     P_RE->load + P_dis + P_grid->load + P_unmet = P_adj
    grid limit: P_grid->load + P_ch,grid <= P_grid_max
    storage:    E_{t+1} = E_t + eta_ch (P_ch,RE + P_ch,grid) dt - P_dis dt / eta_dis  (kWh)

Storage provenance: the stored energy is split into a renewable-origin pool and a
grid-origin pool. Charging adds to the pool of its source; discharging draws from the
two pools in proportion to their current shares (well-mixed store). The initial
content is counted as grid-origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, vstack

DT = 0.25
STEPS_PER_YEAR = 35_040

BASE = {
    "battery_kwh": 20.0, "soc_min": 0.20, "soc_max": 0.90, "soc0": 0.50,
    "p_ch": 5.0, "p_dis": 5.0, "eta_ch": 0.95, "eta_dis": 0.95,
    "grid_max": 14.0,
    "grid_charging": True, "grid_charge_target": 0.60,
    "dr_cap": 0.15, "dr_elasticity": 0.30, "dr_rebound": 0.50,
    "c_deg": 0.015, "c_bess_fixed": 800.0, "c_dr": 0.05, "c_dr_fixed": 150.0,
    "c_unmet": 5.0, "fixed_cost_mult": 1.0,
}
POLICIES = ["A1", "A2", "A3", "A4"]
HARDWARE = {"A1": (False, False), "A2": (True, False), "A3": (False, True), "A4": (True, True)}
SHORT = {"A1": "Renewable-first", "A2": "Battery", "A3": "Price-elastic DR", "A4": "Hybrid",
         "L2": "LP, battery", "L3": "LP, DR", "L4": "LP, battery + DR"}


def dr_fraction(price: np.ndarray, p: dict) -> np.ndarray:
    pibar = float(np.mean(price))
    return np.minimum(p["dr_cap"], p["dr_elasticity"] * np.maximum(0.0, (price - pibar) / pibar))


def fixed_cost_per_step(has_bat: bool, has_dr: bool, p: dict) -> float:
    annual = (p["c_bess_fixed"] if has_bat else 0.0) + (p["c_dr_fixed"] if has_dr else 0.0)
    return annual * p["fixed_cost_mult"] / STEPS_PER_YEAR


def simulate(df: pd.DataFrame, policy: str, p: dict) -> pd.DataFrame:
    has_bat, has_dr = HARDWARE[policy]
    n = len(df)
    load = df["load_kw"].to_numpy(); re = (df["pv_kw"] + df["wind_kw"]).to_numpy()
    price = df["price"].to_numpy(); hour = df.index.hour.to_numpy(); minute = df.index.minute.to_numpy()
    onpeak = (df["period"] == "t3_onpeak").to_numpy(); offpeak = (df["period"] == "t1_offpeak").to_numpy()
    rho = dr_fraction(price, p) if has_dr else np.zeros(n)
    Emax = p["battery_kwh"] * p["soc_max"]; Emin = p["battery_kwh"] * p["soc_min"]
    Etarget = p["battery_kwh"] * p["grid_charge_target"]
    E = p["battery_kwh"] * p["soc0"] if has_bat else 0.0
    E_re = 0.0                                                     # renewable-origin content
    gmax = p["grid_max"]; ec, ed = p["eta_ch"], p["eta_dis"]
    keys = ["req", "dr", "adj", "re_load", "ch_re", "ch_g", "dis", "dis_re", "g_load", "unmet", "curt", "soc"]
    cols = {k: np.zeros(n) for k in keys}
    deferred, rebound_rate = 0.0, 0.0
    for t in range(n):
        if hour[t] == 23 and minute[t] == 0:
            rebound_rate = deferred / (32 * DT); deferred = 0.0
        req = load[t] + (rebound_rate if offpeak[t] else 0.0)
        dr = rho[t] * load[t]
        deferred += p["dr_rebound"] * dr * DT
        adj = req - dr
        re_load = min(re[t], adj); deficit = adj - re_load; surplus = re[t] - re_load
        ch_re = ch_g = dis = dis_re = 0.0
        if has_bat:
            ch_re = max(0.0, min(surplus, p["p_ch"], (Emax - E) / (ec * DT)))
            E += ec * ch_re * DT; E_re += ec * ch_re * DT; surplus -= ch_re
            need = deficit if onpeak[t] else max(0.0, deficit - gmax)
            if need > 0:
                dis = max(0.0, min(need, p["p_dis"], ed * (E - Emin) / DT))
                if dis > 0:
                    share = E_re / E if E > 1e-12 else 0.0
                    drawn = dis * DT / ed
                    E -= drawn; E_re -= share * drawn; dis_re = share * dis; deficit -= dis
        g_load = min(deficit, gmax); unmet = deficit - g_load
        if has_bat and p["grid_charging"] and offpeak[t] and E < Etarget:
            ch_g = max(0.0, min(p["p_ch"] - ch_re, gmax - g_load, (Etarget - E) / (ec * DT)))
            E += ec * ch_g * DT
        E_re = min(max(E_re, 0.0), E)
        vals = (req, dr, adj, re_load, ch_re, ch_g, dis, dis_re, g_load, unmet, surplus, E)
        for k, v in zip(keys, vals):
            cols[k][t] = v
    out = pd.DataFrame(cols, index=df.index)
    return _finish(out, df, has_bat, has_dr, p, policy)


def _finish(out, df, has_bat, has_dr, p, label):
    out["re"] = (df["pv_kw"] + df["wind_kw"]).to_numpy()
    out["load"] = df["load_kw"].to_numpy(); out["price"] = df["price"].to_numpy()
    out["grid"] = out["g_load"] + out["ch_g"]
    out["energy_cost"] = (out["price"] * out["grid"] + p["c_deg"] * (out["ch_re"] + out["ch_g"] + out["dis"])
                          + p["c_dr"] * out["dr"]) * DT + fixed_cost_per_step(has_bat, has_dr, p)
    out["penalty"] = p["c_unmet"] * out["unmet"] * DT
    out["cost"] = out["energy_cost"] + out["penalty"]
    out["policy"] = label
    assert np.allclose(out.re_load + out.ch_re + out.curt, out.re, atol=1e-7)
    assert np.allclose(out.re_load + out.dis + out.g_load + out.unmet, out.adj, atol=1e-7)
    assert (out.grid <= p["grid_max"] + 1e-7).all()
    assert (out.dis_re <= out.dis + 1e-9).all()
    return out


def provenance(out: pd.DataFrame, p: dict) -> np.ndarray:
    """Renewable-origin share of discharge for given charge/discharge flows (well-mixed store)."""
    E = p["battery_kwh"] * p["soc0"]; E_re = 0.0; ec, ed = p["eta_ch"], p["eta_dis"]
    ch_re = out["ch_re"].to_numpy(); ch_g = out["ch_g"].to_numpy(); dis = out["dis"].to_numpy()
    dis_re = np.zeros(len(out))
    for t in range(len(out)):
        E += ec * ch_re[t] * DT; E_re += ec * ch_re[t] * DT
        if dis[t] > 0:
            share = E_re / E if E > 1e-12 else 0.0
            drawn = dis[t] * DT / ed
            E -= drawn; E_re -= share * drawn; dis_re[t] = share * dis[t]
        E += ec * ch_g[t] * DT
        E_re = min(max(E_re, 0.0), max(E, 0.0))
    return dis_re


def lp_benchmark(df: pd.DataFrame, p: dict, has_bat=True, has_dr=True, label="L4") -> pd.DataFrame:
    """Perfect-foresight daily LP (windows start at 23:00) for the given hardware."""
    n_all = len(df)
    load = df["load_kw"].to_numpy(); re = (df["pv_kw"] + df["wind_kw"]).to_numpy()
    price = df["price"].to_numpy(); hour = df.index.hour.to_numpy(); minute = df.index.minute.to_numpy()
    offpeak = (df["period"] == "t1_offpeak").to_numpy()
    rho = dr_fraction(price, p) if has_dr else np.zeros(n_all)
    starts = [0] + [t for t in range(1, n_all) if hour[t] == 23 and minute[t] == 0] + [n_all]
    bk = p["battery_kwh"] if has_bat else 0.0
    Emax = bk * p["soc_max"]; Emin = bk * p["soc_min"]; E0 = bk * p["soc0"]
    pch = p["p_ch"] if has_bat else 0.0; pdis = p["p_dis"] if has_bat else 0.0
    keys = ["req", "dr", "adj", "re_load", "ch_re", "ch_g", "dis", "g_load", "unmet", "curt", "soc"]
    res = {k: np.zeros(n_all) for k in keys}
    carry = 0.0
    V = 8
    for a, b in zip(starts[:-1], starts[1:]):
        n = b - a; idx = np.arange(a, b)
        reb = np.where(offpeak[idx], carry / (32 * DT), 0.0) if a > 0 else np.zeros(n)
        req = load[idx] + reb
        nv = V * n + (n + 1)
        vi = lambda k, t: k * n + t
        ei = lambda t: V * n + t
        c = np.zeros(nv)
        for t in range(n):
            pi = price[idx[t]]
            c[vi(2, t)] = pi * DT + p["c_deg"] * DT; c[vi(4, t)] = pi * DT
            c[vi(1, t)] = p["c_deg"] * DT; c[vi(3, t)] = p["c_deg"] * DT
            c[vi(6, t)] = p["c_unmet"] * DT; c[vi(7, t)] = p["c_dr"] * DT; c[vi(5, t)] = 1e-6
        Aeq = lil_matrix((3 * n + 1, nv)); beq = np.zeros(3 * n + 1)
        Aub = lil_matrix((2 * n + 1, nv)); bub = np.zeros(2 * n + 1)
        for t in range(n):
            Aeq[t, vi(0, t)] = 1; Aeq[t, vi(1, t)] = 1; Aeq[t, vi(5, t)] = 1; beq[t] = re[idx[t]]
            r = n + t
            for k in (0, 3, 4, 6, 7):
                Aeq[r, vi(k, t)] = 1
            beq[r] = req[t]
            r = 2 * n + t
            Aeq[r, ei(t + 1)] = 1; Aeq[r, ei(t)] = -1
            Aeq[r, vi(1, t)] = -p["eta_ch"] * DT; Aeq[r, vi(2, t)] = -p["eta_ch"] * DT
            Aeq[r, vi(3, t)] = DT / p["eta_dis"]
            Aub[t, vi(4, t)] = 1; Aub[t, vi(2, t)] = 1; bub[t] = p["grid_max"]
            Aub[n + t, vi(1, t)] = 1; Aub[n + t, vi(2, t)] = 1; bub[n + t] = pch
        Aeq[3 * n, ei(0)] = 1; beq[3 * n] = E0
        Aub[2 * n, ei(n)] = -1; Aub[2 * n, ei(0)] = 1                 # no net depletion
        bounds = [(0, None)] * (V * n) + [(Emin, Emax)] * (n + 1)
        for t in range(n):
            bounds[vi(3, t)] = (0, pdis); bounds[vi(7, t)] = (0, rho[idx[t]] * load[idx[t]])
            if not has_bat:
                bounds[vi(1, t)] = (0, 0); bounds[vi(2, t)] = (0, 0)
        sol = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
        if sol.status != 0:
            raise RuntimeError(f"LP failed at {df.index[a]}: {sol.message}")
        x = sol.x; get = lambda k: x[k * n:(k + 1) * n]
        res["re_load"][idx] = get(0); res["ch_re"][idx] = get(1); res["ch_g"][idx] = get(2)
        res["dis"][idx] = get(3); res["g_load"][idx] = get(4); res["curt"][idx] = get(5)
        res["unmet"][idx] = get(6); res["dr"][idx] = get(7); res["req"][idx] = req
        res["adj"][idx] = req - get(7); res["soc"][idx] = x[V * n + 1: V * n + n + 1]
        E0 = x[V * n + n]
        carry = p["dr_rebound"] * get(7).sum() * DT
    out = pd.DataFrame(res, index=df.index)
    for k in ["re_load", "ch_re", "ch_g", "dis", "g_load", "curt", "unmet", "dr"]:
        out[k] = out[k].clip(lower=0)
    re_av = (df["pv_kw"] + df["wind_kw"]).to_numpy()
    out["re_load"] = np.minimum(out["re_load"], re_av)
    out["curt"] = (re_av - out["re_load"] - out["ch_re"]).clip(lower=0)
    out["unmet"] = (out["adj"] - out["re_load"] - out["dis"] - out["g_load"]).clip(lower=0)
    pp = dict(p); pp["battery_kwh"] = bk if bk > 0 else 1e-9
    out["dis_re"] = provenance(out, pp) if has_bat else 0.0
    return _finish(out, df, has_bat, has_dr, p, label)


# ---------------------------------------------------------------- criteria
CRITERIA = ["C1", "C2", "C3", "C4", "C5"]
CRIT_NAMES = {"C1": "Cost efficiency (per served kWh)", "C2": "Supply reliability",
              "C3": "Renewable self-consumption", "C4": "Renewable-origin supply share",
              "C5": "Peak grid-import headroom"}
BLOCK_LEN = {"t1_offpeak": 32, "t2_midpeak": 40, "t3_onpeak": 24}


def block_keys(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Operating day runs 07:00 -> 07:00; each has one block per TOU period."""
    return (index - pd.Timedelta(hours=7)).normalize()


def block_criteria(sim: pd.DataFrame, df: pd.DataFrame, p: dict) -> pd.DataFrame:
    g = sim.copy()
    g["period"] = df["period"].to_numpy(); g["opday"] = block_keys(df.index)
    gmax = p["grid_max"]
    ref_imp = np.minimum(g["load"], gmax)
    g["ref_energy_cost"] = g["price"] * ref_imp * DT
    g["ref_served"] = ref_imp * DT
    g["ref_penalty"] = p["c_unmet"] * np.maximum(0, g["load"] - gmax) * DT
    g["served"] = (g["adj"] - g["unmet"]) * DT
    agg = g.groupby(["opday", "period"]).agg(
        n=("cost", "size"), cost=("cost", "sum"), energy_cost=("energy_cost", "sum"),
        ref_energy_cost=("ref_energy_cost", "sum"), ref_served=("ref_served", "sum"),
        ref_penalty=("ref_penalty", "sum"), served=("served", "sum"),
        unmet=("unmet", "sum"), adj=("adj", "sum"), re=("re", "sum"), re_load=("re_load", "sum"),
        ch_re=("ch_re", "sum"), dis=("dis", "sum"), dis_re=("dis_re", "sum"), grid=("grid", "sum"),
        peak=("grid", "max"), curt=("curt", "sum"))
    agg = agg[agg["n"] == agg.index.get_level_values("period").map(BLOCK_LEN)]
    out = pd.DataFrame(index=agg.index)
    y = agg["energy_cost"] / agg["served"]                         # USD per served kWh
    rho = agg["ref_energy_cost"] / agg["ref_served"]
    out["C1"] = rho / (rho + y)
    ref_tot = agg["ref_energy_cost"] + agg["ref_penalty"]
    out["C1_pen"] = ref_tot / (ref_tot + agg["cost"])               # total cost incl. penalty
    out["C1_tot"] = agg["ref_energy_cost"] / (agg["ref_energy_cost"] + agg["energy_cost"])
    out["C1_lin"] = (1 - y / rho).clip(0, 1)                        # linear, clipped
    out["C2"] = 1 - agg["unmet"] / agg["adj"]
    out["C3"] = np.where(agg["re"] > 1e-6, (agg["re_load"] + agg["ch_re"]) / agg["re"].where(agg["re"] > 1e-6, 1), 1.0)
    out["C4"] = (agg["re_load"] + agg["dis_re"]) / agg["adj"]
    out["C4_local"] = (agg["re_load"] + agg["dis"]) / agg["adj"]
    out["C5"] = 1 - agg["peak"] / gmax
    for k in ["cost", "energy_cost", "served", "unmet", "adj", "re", "re_load", "ch_re", "dis", "dis_re",
              "grid", "curt"]:
        out[k] = agg[k] * (1.0 if k in ("cost", "energy_cost", "served") else DT)
    out["peak"] = agg["peak"]
    return out.reset_index()


def period_totals(blocks: pd.DataFrame) -> pd.DataFrame:
    g = blocks.groupby("period")
    return pd.DataFrame({
        "energy_cost_usd": g["energy_cost"].sum(), "served_kwh": g["served"].sum(),
        "unmet_kwh": g["unmet"].sum(), "adj_kwh": g["adj"].sum(), "re_kwh": g["re"].sum(),
        "re_used_kwh": g["re_load"].sum() + g["ch_re"].sum(), "re_origin_kwh": g["re_load"].sum() + g["dis_re"].sum(),
        "grid_kwh": g["grid"].sum(), "mean_block_peak_kw": g["peak"].mean(), "curt_kwh": g["curt"].sum()})
