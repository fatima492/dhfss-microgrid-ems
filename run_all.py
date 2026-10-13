"""Reproduce every table and figure of the manuscript and its supplement.

    python src/gen_household_load.py   # only if data/household_load_15min.csv is missing (~15 min)
    python run_all.py                  # ~35-45 min (global sensitivity analysis ~25 min)
    python run_all.py --skip-gsa       # reuse results/gsa_morris.csv

Outputs: results/*.csv, results/summary.json, results/tables/*.tex, figures/*.png
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import warnings

# --- dependency check: a clear message instead of a ModuleNotFoundError deep inside the pipeline ---
_REQUIRED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "matplotlib": "matplotlib",
             "pvlib": "pvlib", "demandlib": "demandlib", "SALib": "SALib"}
_missing = [pkg for mod, pkg in _REQUIRED.items() if importlib.util.find_spec(mod) is None]
if _missing:
    _exe = sys.executable
    sys.stderr.write(
        "\nMissing Python package(s): " + ", ".join(_missing) + "\n"
        "This script is running with the Python at:\n    " + _exe + "\n"
        "Install the packages into exactly this Python by running, in a terminal:\n\n"
        '    "' + _exe + '" -m pip install ' + " ".join(_missing) + "\n\n"
        "or, from the bundle folder, install everything:\n\n"
        '    "' + _exe + '" -m pip install -r requirements.txt\n\n'
        "In Spyder or Jupyter, type  %pip install " + " ".join(_missing) + "  in the console, then restart the kernel.\n"
        "See README.md, section 'Quick start'.\n\n")
    sys.exit(1)
for _stream in (sys.stdout, sys.stderr):          # Windows consoles default to a legacy code page
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import ems_data as E  # noqa: E402
import ems_model as M  # noqa: E402
import mcdm as D  # noqa: E402
import pipeline as P  # noqa: E402

SEED = 2026
RES = os.path.join(HERE, "results"); TAB = os.path.join(RES, "tables"); FIG = os.path.join(HERE, "figures")
for d in (RES, TAB, FIG):
    os.makedirs(d, exist_ok=True)
A = P.ALTS
BASE_PARAMS = {}                                  # M.BASE already holds the baseline (grid limit 14 kW)
OM = P.OMEGA_DURATION
W0 = P.W_EQUAL
OUT = {}


def rk(s):
    parts = [x.strip() for x in str(s).split(">") if x.strip()]
    if not parts:
        return "none feasible"
    return "$" + "{\\succ}".join(x[0] + "_" + x[1:] for x in parts) + "$"


def f4(x):
    return f"{x:.4f}"


def save_tex(name, lines):
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def mabac_rank(S, w=W0, labels=A):
    Q = D.mabac(S, w)[0]
    return Q, D.ranking_string(Q, labels)


# ============================================================== data and baseline
def section_data():
    df = P.inputs("stochastic")
    diag = E.diagnostics(df[df.index.year.isin(E.TEST_YEARS)])
    rows = {}
    for lm in ["stochastic", "h0", "metered"]:
        d = P.inputs(lm); d = d[d.index.year.isin(E.TEST_YEARS)]
        g = E.diagnostics(d)
        rows[lm] = {"peak_hour": int(g["hourly"]["load_kw"].idxmax()),
                    "peak_to_mean": float(g["hourly"]["load_kw"].max() / g["hourly"]["load_kw"].mean()),
                    "daily_cv": g["daily_cv"], "daily_resid_cv": g["daily_resid_cv"],
                    "acf1": g["acf"]["load_kw"][1], "acf96": g["acf"]["load_kw"][96],
                    "max_kw": float(d.load_kw.max()),
                    **{f"mean_{k}": float(v) for k, v in g["period_means"]["load_kw"].items()}}
    lm_tab = pd.DataFrame(rows).T; lm_tab.to_csv(os.path.join(RES, "load_models.csv"))
    OUT["load_models"] = lm_tab.round(4).to_dict(orient="index")
    full = E.diagnostics(df)
    full["annual"].to_csv(os.path.join(RES, "annual_inputs.csv"))
    OUT["annual_inputs"] = full["annual"].round(1).to_dict(orient="index")
    OUT["period_means_test"] = diag["period_means"].round(3).to_dict()
    OUT["acf_test"] = diag["acf"]
    OUT["pv_night_share"] = full["pv_night_share"]
    names = {"stochastic": "Stochastic neighbourhood (baseline)", "h0": "BDEW H0 standard profile",
             "metered": "Metered low-voltage aggregate"}
    save_tex("load_models", [
        f"{names[k]} & {int(r['peak_hour']):d}:00 & {r['peak_to_mean']:.2f} & {r['mean_t1_offpeak']:.2f} & "
        f"{r['mean_t2_midpeak']:.2f} & {r['mean_t3_onpeak']:.2f} & {100*r['daily_resid_cv']:.1f} & {r['acf1']:.3f} \\\\"
        for k, r in lm_tab.iterrows()])
    a = full["annual"]
    save_tex("annual_inputs", [f"{y} & {r.ghi_kwh_m2:,.0f} & {r.pv_kwh:,.0f} & {r.wind_kwh:,.0f} & {r.load_kwh:,.0f} \\\\"
                               for y, r in a.iterrows()])
    return df


def section_baseline():
    R = P.run(BASE_PARAMS, with_lp=True)
    OUT["targets"] = R["targets"]
    R["tune_table"].to_csv(os.path.join(RES, "tuning_train.csv"), index=False)
    tt = R["tune_table"].pivot(index="target", columns="policy", values="train_cost_usd")
    tt = tt.loc[P.GRID_TARGETS]
    def tl(t):
        return "off" if t == "off" else "%.0f\\%%" % (100 * float(t))
    save_tex("tuning", [tl(t) + f" & {r['A2']/3:,.0f} & {r['A4']/3:,.0f} \\\\" for t, r in tt.iterrows()])
    # KPIs (annual averages over the test years)
    ny = len(E.TEST_YEARS); rows = []
    for a, s in R["sims"].items():
        rows.append({"policy": a, "cost_usd": s.cost.sum() / ny, "energy_cost_usd": s.energy_cost.sum() / ny,
                     "unmet_kwh": s.unmet.sum() * M.DT / ny, "grid_kwh": s.grid.sum() * M.DT / ny,
                     "grid_charge_kwh": s.ch_g.sum() * M.DT / ny, "discharge_kwh": s.dis.sum() * M.DT / ny,
                     "discharge_re_kwh": s.dis_re.sum() * M.DT / ny, "curt_kwh": s.curt.sum() * M.DT / ny,
                     "dr_kwh": s.dr.sum() * M.DT / ny,
                     "reliability": 1 - s.unmet.sum() / s.adj.sum(),
                     "self_consumption": (s.re_load.sum() + s.ch_re.sum()) / s.re.sum(),
                     "cost_per_served": s.energy_cost.sum() / ((s.adj - s.unmet).sum() * M.DT)})
    kpi = pd.DataFrame(rows).set_index("policy"); kpi.to_csv(os.path.join(RES, "kpi_test.csv"))
    OUT["kpi"] = kpi.round(4).to_dict(orient="index")
    save_tex("kpi", [f"{a} & {M.SHORT[a]} & {r.energy_cost_usd:,.0f} & {100*r.cost_per_served:.2f} & {r.unmet_kwh:,.0f} & "
                     f"{100*r.reliability:.2f} & {r.grid_charge_kwh:,.0f} & {r.discharge_kwh:,.0f} & "
                     f"{(100*r.discharge_re_kwh/r.discharge_kwh if r.discharge_kwh > 0 else float('nan')):.0f} \\\\"
                     .replace("nan", "--") for a, r in kpi.iterrows()])
    return R


def section_scores(R):
    Q, S, X, days = P.base_Q(R["blocks"])
    OUT["n_days"] = len(days); OUT["S"] = S.round(4).tolist(); OUT["Q"] = Q.round(4).tolist()
    OUT["ranking"] = D.ranking_string(Q, A)
    Qv, V, g, Qm = D.mabac(S, W0)
    rnk = D.ranks(Q)
    H = D.build_hfes(X, "quantile")
    _, _, Dv = D.aggregate_matrix(H, OM, "IND", "AR")
    pd.DataFrame(S, index=A, columns=M.CRITERIA).to_csv(os.path.join(RES, "scores.csv"))
    pd.DataFrame(Qm, index=A, columns=M.CRITERIA).to_csv(os.path.join(RES, "mabac_distances.csv"))
    save_tex("mabac", [f"{a} & " + " & ".join(f"{S[i, j]:.4f}" for j in range(5)) + f" & {Q[i]:.4f} & {rnk[i]} \\\\"
                       for i, a in enumerate(A)] + ["\\midrule", "$g_j$ (on $v_{ij}$) & " + " & ".join(f"{x:.4f}" for x in g) + " & & \\\\"])
    # HFE tables (supplement) and correlation
    for k, per in enumerate(P.PERIODS):
        save_tex(f"hfe_{per}", [f"{a} & " + " & ".join("\\{" + ",".join(f"{v:.3f}" for v in H[i][j][k]) + "\\}"
                                                     for j in range(5)) + " \\\\" for i, a in enumerate(A)])
    pooled = pd.concat([R["blocks"][a][M.CRITERIA] for a in A]); corr = pooled.corr()
    corr.to_csv(os.path.join(RES, "criterion_correlation.csv"))
    save_tex("criterion_corr", [f"$C_{c[1]}$ & " + " & ".join(f"{corr.loc[c, d]:.2f}" for d in M.CRITERIA) + " \\\\"
                                for c in M.CRITERIA])
    OUT["max_offdiag_corr"] = float(np.max(np.abs(corr.values - np.eye(5))))
    OUT["degenerate_hfes"] = int(sum(np.ptp(H[i][j][k]) < 1e-9 for i in range(4) for j in range(5) for k in range(3)))
    OUT["hfes_with_one"] = int(sum(np.isclose(H[i][j][k], 1).any() for i in range(4) for j in range(5) for k in range(3)))
    np.save(os.path.join(RES, "block_tensor_test.npy"), X)
    pd.Series([str(d.date()) for d in days]).to_csv(os.path.join(RES, "block_days.csv"), index=False, header=["opday"])
    return Q, S, X, days, H, Dv


# ============================================================== context-dependent parameter sets (DHFSS)
def section_context(H):
    """Dynamic hesitant fuzzy soft set with context-dependent active criteria.

    Baseline: every criterion is active in every period (A_t = E).  Variant: renewable
    self-consumption C3 is inactive off-peak and on-peak, where little renewable energy is
    available and the ratio is ill-defined (A_t1 = A_t3 = E minus C3, A_t2 = E).  Temporal
    weights are renormalized over the periods in which a criterion is active."""
    m, n, p = len(A), len(M.CRITERIA), len(P.PERIODS)
    Sk = np.array([[[np.mean(H[i][j][k]) for k in range(p)] for j in range(n)] for i in range(m)])
    rows, lines = [], []
    for label, inactive in (("All criteria active in every period ($A_t=\\mathcal{C}$, baseline)", []),
                            ("$C_3$ inactive off-peak and on-peak ($A_{t_1}=A_{t_3}=\\mathcal{C}\\setminus\\{C_3\\}$)", [(2, 0), (2, 2)])):
        act = np.ones((n, p), bool)
        for j, k in inactive:
            act[j, k] = False
        S = np.zeros((m, n))
        for j in range(n):
            w = np.asarray(OM) * act[j]; S[:, j] = Sk[:, j, :] @ (w / w.sum())
        Q = D.mabac(S, W0)[0]
        rows.append({"variant": label, **{a: Q[i] for i, a in enumerate(A)}, "C3_scores": S[:, 2].round(4).tolist(),
                     "ranking": D.ranking_string(Q, A)})
        lines.append(f"{label} & " + " & ".join(f4(q) for q in Q) + " & " + rk(D.ranking_string(Q, A)) + " \\\\")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "context_parameters.csv"), index=False)
    OUT["context_dhfss"] = [{k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in r.items()} for r in rows]
    save_tex("context", lines)


# ============================================================== criterion definitions
def section_criteria(R):
    rows = []
    variants = [("C1 per served kWh, no penalty; C4 renewable origin (baseline)", ["C1", "C2", "C3", "C4", "C5"]),
                ("C1 total cost, no penalty", ["C1_tot", "C2", "C3", "C4", "C5"]),
                ("C1 total cost incl. unserved-energy penalty", ["C1_pen", "C2", "C3", "C4", "C5"]),
                ("C1 linear-clipped transform $1-y/\\rho$", ["C1_lin", "C2", "C3", "C4", "C5"]),
                ("C4 local supply incl. grid-charged storage", ["C1", "C2", "C3", "C4_local", "C5"]),
                ("C1 with penalty and C4 local (both comparators)", ["C1_pen", "C2", "C3", "C4_local", "C5"])]
    for nm, cr in variants:
        Q, _, _, _ = P.base_Q(R["blocks"], criteria=cr)
        rows.append({"variant": nm, **{a: Q[i] for i, a in enumerate(A)}, "ranking": D.ranking_string(Q, A)})
    for c4 in ["C4", "C4_local"]:
        cr = ["C1", "C2", "C3", c4, "C5"]
        _, S, _, _ = P.base_Q(R["blocks"], criteria=cr)
        for j in range(5):
            w = W0.copy(); w[j] = 0; w /= w.sum(); Q = D.mabac(S, w)[0]
            rows.append({"variant": f"leave out $C_{j+1}$ ({'renewable-origin' if c4 == 'C4' else 'local'} $C_4$)",
                         **{a: Q[i] for i, a in enumerate(A)}, "ranking": D.ranking_string(Q, A)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "criterion_variants.csv"), index=False)
    OUT["criterion_variants"] = t.round(4).to_dict(orient="records")
    save_tex("criterion_variants", [f"{r['variant']} & " + " & ".join(f4(r[a]) for a in A) + " & " + rk(r["ranking"]) + " \\\\"
                                    + ("\n\\midrule" if i == 5 else "") for i, r in t.iterrows()])
    # reliability as a constraint
    rel = {a: 1 - R["sims"][a].unmet.sum() / R["sims"][a].adj.sum() for a in A}
    OUT["reliability_test"] = rel
    _, S, _, _ = P.base_Q(R["blocks"])
    rows = []
    for thr in [0.985, 0.990, 0.995, 0.9975]:
        keep = [i for i, a in enumerate(A) if rel[a] >= thr]
        if len(keep) >= 2:
            S2 = S[keep][:, [0, 2, 3, 4]]; w = np.full(4, 0.25)
            Q = D.mabac(S2, w)[0]; rnk = D.ranking_string(Q, [A[i] for i in keep])
        else:
            rnk = " > ".join(A[i] for i in keep)
        rows.append({"threshold": thr, "feasible": ", ".join(A[i] for i in keep), "ranking": rnk})
    t2 = pd.DataFrame(rows); t2.to_csv(os.path.join(RES, "reliability_constraint.csv"), index=False)
    OUT["reliability_constraint"] = t2.to_dict(orient="records")
    save_tex("reliability_constraint", [f"{100*r['threshold']:.2f}\\% & {r['feasible']} & {rk(r['ranking'])} \\\\"
                                        for _, r in t2.iterrows()])


# ============================================================== hesitancy, operators, methods
def section_hesitancy(X, S):
    rows = []
    for kap in [0, 0.25, 0.5, 1.0, 1.5, 2.0]:
        H = D.build_hfes(X, "meansd", kap); _, Sk, _ = D.aggregate_matrix(H, OM)
        Q, r = mabac_rank(Sk); rows.append({"construction": f"meansd:{kap}", **dict(zip(A, Q)), "ranking": r})
    Hb = D.build_hfes(X, "bootstrap", rng=np.random.default_rng(SEED)); _, Sb, _ = D.aggregate_matrix(Hb, OM)
    Q, r = mabac_rank(Sb); rows.append({"construction": "bootstrap", **dict(zip(A, Q)), "ranking": r})
    Q, r = mabac_rank(S); rows.append({"construction": "quantile", **dict(zip(A, Q)), "ranking": r})
    Q, r = mabac_rank(D.scores_quantile(X, OM, qs=np.linspace(0.05, 0.95, 19))); rows.append({"construction": "quantile19", **dict(zip(A, Q)), "ranking": r})
    t = pd.DataFrame(rows); crisp = t.iloc[0][A].to_numpy(float)
    t["max_abs_dQ"] = [float(np.max(np.abs(r_[A].to_numpy(float) - crisp))) for _, r_ in t.iterrows()]
    t.to_csv(os.path.join(RES, "hfe_construction.csv"), index=False)
    OUT["hfe_construction"] = t.round(5).to_dict(orient="records")
    OUT["max_dQ_hesitancy"] = float(t["max_abs_dQ"].max())
    def nm(c):
        if c == "meansd:0":
            return "Crisp daily mean ($\\kappa=0$)"
        if c.startswith("meansd"):
            return "$\\{\\mu-\\kappa\\sigma,\\mu,\\mu+\\kappa\\sigma\\}$, $\\kappa=" + c.split(":")[1] + "$"
        return {"bootstrap": "Bootstrap-mean $\\{P_{10},P_{50},P_{90}\\}$",
                "quantile": "Daily-block $\\{P_{10},P_{50},P_{90}\\}$ (baseline)",
                "quantile19": "Daily-block $\\{P_{5},P_{10},\\dots,P_{95}\\}$"}[c]
    save_tex("hfe_construction", [f"{nm(r['construction'])} & " + " & ".join(f4(r[a]) for a in A) +
                                  f" & {r['max_abs_dQ']:.4f} & " + rk(r["ranking"]) + " \\\\" for _, r in t.iterrows()])


def section_operators(X, H):
    rows = []
    for coup in ["IND", "COM"]:
        for mean in ["AR", "EIN", "ALG"]:
            for eps in ([0.0] if mean == "AR" else [0.0, 1e-3, 1e-2]):
                Hagg, Sx, _ = D.aggregate_matrix(H, OM, coup, mean, eps)
                Q = D.mabac(Sx, W0)[0]
                rows.append({"operator": f"{coup}-{mean}", "eps": eps, "method": "MABAC", **dict(zip(A, Q)),
                             "ranking": D.ranking_string(Q, A), "S": Sx})
    Hagg, Sx, _ = D.aggregate_matrix(H, OM, "IND", "AR")
    for meth, vals in [("WSM", D.wsm(Sx, W0)), ("TOPSIS", D.topsis(Sx, W0)[0]), ("HF-TOPSIS", D.hf_topsis(Hagg, W0)[0])]:
        rows.append({"operator": "IND-AR", "eps": 0.0, "method": meth, **dict(zip(A, vals)),
                     "ranking": D.ranking_string(vals, A), "S": Sx})
    t = pd.DataFrame(rows)
    t.drop(columns=["S"]).to_csv(os.path.join(RES, "operators_methods.csv"), index=False)
    OUT["operators_methods"] = t.drop(columns=["S"]).round(5).to_dict(orient="records")
    def label(r):
        cpl = {"IND": "\\perp", "COM": "\\top"}[r["operator"].split("-")[0]]
        mean = r["operator"].split("-")[1]
        s = f"$\\mathrm{{THFWA}}^{{{cpl}}}_{{\\mathrm{{{mean}}}}}$"
        if r["eps"] > 0:
            s += f", $\\varepsilon={r['eps']:g}$"
        return s + " + " + r["method"]
    lines = []
    for _, r in t.iterrows():
        rr = D.ranks([r[a] for a in A])
        lines.append(f"{label(r)} & " + " & ".join(str(x) for x in rr) + " & " + rk(r["ranking"]) + " \\\\")
    save_tex("operators_ranks", lines)
    sel = t[(t.eps == 0.0)]
    lines = []
    for _, r in sel.iterrows():
        lines.append(f"{label(r)} & " + " & ".join(f4(r[a]) for a in A) + " \\\\")
    save_tex("operators_scores", lines)
    # absorbing: share of period-level HFEs that contain a grade equal to 1, per criterion
    OUT["absorbing_by_criterion"] = {M.CRITERIA[j]: int(sum(np.isclose(H[i][j][k], 1).any() for i in range(4) for k in range(3)))
                                     for j in range(5)}
    _, Se, _ = D.aggregate_matrix(H, OM, "IND", "EIN", 0.0)
    OUT["absorbed_cells_EIN"] = int(np.isclose(Se, 1.0).sum())


def raw_period_totals(blocks):
    tot = {a: M.period_totals(blocks[a]).reindex(P.PERIODS) for a in A}
    def raw(t):
        return np.stack([t.energy_cost_usd / t.served_kwh, 1 - t.unmet_kwh / t.adj_kwh, t.re_used_kwh / t.re_kwh,
                         t.re_origin_kwh / t.adj_kwh, t.mean_block_peak_kw], axis=0)
    return np.stack([raw(tot[a]) for a in A]), tot


def section_normalization(R, S):
    p = R["p"]; rows = []
    Q, r = mabac_rank(S); rows.append({"scheme": "N1", **dict(zip(A, Q)), "ranking": r})
    Rr, tot = raw_period_totals(R["blocks"])
    refc = np.zeros((4, 3))
    for i, a in enumerate(A):
        b = R["blocks"][a]
        for k, per in enumerate(P.PERIODS):
            sub = b[b.period == per]
            ref_cost = (sub["ref_energy_cost"].sum() if "ref_energy_cost" in sub else np.nan)
            refc[i, k] = ref_cost
    # reference cost per served kWh of the no-asset system, per period (identical for all alternatives)
    df = R["df"]; gmax = p["grid_max"]
    imp = np.minimum(df.load_kw, gmax)
    ref = (df.assign(c=df.price * imp, s=imp).groupby("period")[["c", "s"]].sum()).reindex(P.PERIODS)
    rho = (ref.c / ref.s).to_numpy()
    Z = Rr.copy(); Z[:, 0, :] = rho / (rho + Rr[:, 0, :]); Z[:, 4, :] = 1 - Rr[:, 4, :] / gmax
    Q, r = mabac_rank(np.tensordot(Z, OM, axes=([2], [0]))); rows.append({"scheme": "N1c", **dict(zip(A, Q)), "ranking": r})
    Z = np.zeros_like(Rr)
    for j in range(5):
        for k in range(3):
            v = Rr[:, j, k]; lo, hi = v.min(), v.max(); cost = j in (0, 4)
            Z[:, j, k] = 0.5 if np.isclose(lo, hi) else ((hi - v) / (hi - lo) if cost else (v - lo) / (hi - lo))
    Q, r = mabac_rank(np.tensordot(Z, OM, axes=([2], [0]))); rows.append({"scheme": "N2", **dict(zip(A, Q)), "ranking": r})
    Z = np.zeros_like(Rr)
    for j in range(5):
        for k in range(3):
            v = Rr[:, j, k]; cost = j in (0, 4)
            Z[:, j, k] = (v.min() / v) if cost else (v / v.max() if v.max() > 0 else 1.0)
    Q, r = mabac_rank(np.tensordot(Z, OM, axes=([2], [0]))); rows.append({"scheme": "N3", **dict(zip(A, Q)), "ranking": r})
    Q = interval_minmax(R); rows.append({"scheme": "N4", **dict(zip(A, Q)), "ranking": D.ranking_string(Q, A)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "normalization.csv"), index=False)
    OUT["normalization"] = t.round(5).to_dict(orient="records")
    names = {"N1": "N1 Reference-point, daily-block quantile HFEs (baseline)",
             "N1c": "N1c Reference-point, period totals (crisp)",
             "N2": "N2 Min--max across alternatives, period totals (crisp)",
             "N3": "N3 Linear ratio to best, period totals (crisp)",
             "N4": "N4 Interval-level min--max, ties set to 1, $\\{\\mu\\pm\\sigma,\\mu\\}$ (common practice)"}
    save_tex("normalization", [f"{names[r['scheme']]} & " + " & ".join(f4(r[a]) for a in A) + " & " + rk(r["ranking"]) + " \\\\"
                               for _, r in t.iterrows()])


def interval_minmax(R):
    """Common practice: min-max across alternatives at every 15-min step, ties = 1, mean +- sd HFEs."""
    p = R["p"]; gmax = p["grid_max"]; per = R["df"]["period"].to_numpy()
    C = []
    for a in A:
        s = R["sims"][a]
        with np.errstate(divide="ignore", invalid="ignore"):
            C.append(np.stack([s.energy_cost.to_numpy(), 1 - s.unmet / s.adj, np.where(s.re > 1e-9, (s.re_load + s.ch_re) / s.re, np.nan),
                               (s.re_load + s.dis_re) / s.adj, 1 - s.grid / gmax]))
    C = np.stack(C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo = np.nanmin(C, axis=0); hi = np.nanmax(C, axis=0)
    eq = np.isclose(lo, hi)
    span = np.where(eq, 1.0, hi - lo)
    Z = np.where(np.arange(5)[None, :, None] == 0, (hi - C) / span, (C - lo) / span)
    Z = np.where(eq, 1.0, Z); Z = np.where(np.isnan(C), np.nan, Z)
    S = np.zeros((4, 5))
    for k, pk in enumerate(P.PERIODS):
        msk = per == pk
        for i in range(4):
            for j in range(5):
                z = Z[i, j, msk]; z = z[~np.isnan(z)]
                mu, sd = z.mean(), z.std()
                h = np.unique(np.round(np.clip([mu - sd, mu, mu + sd], 0, 1), 6))
                S[i, j] += OM[k] * h.mean()
    return D.mabac(S, W0)[0]


# ============================================================== coupling and risk
def section_coupling(X, S):
    cells = []
    stats = {}
    for i, a in enumerate(A):
        for j, c in enumerate(M.CRITERIA):
            st = D.coupling_stats(X[i, j], OM, alphas=(0.10, 0.25))
            stats[(i, j)] = st
            sdI, sdE, sdC = st["sd"]["IND"], st["sd"]["EMP"], st["sd"]["COM"]
            pos = (sdE - sdI) / (sdC - sdI) if sdC - sdI > 1e-12 else np.nan
            # same-day weighted covariance sum
            Xk = X[i, j]; cov = np.cov(Xk, bias=True)
            wc = sum(OM[k] * OM[l] * cov[k, l] for k in range(3) for l in range(3) if k < l)
            cells.append({"alt": a, "crit": c, "S": st["S"], "sd_IND": sdI, "sd_EMP": sdE, "sd_COM": sdC,
                          "sd_upper": st["sd_bound_upper"], "pos": pos, "w_cov_sum": wc,
                          "L10_IND": st["L10"]["IND"], "L10_EMP": st["L10"]["EMP"], "L10_COM": st["L10"]["COM"],
                          "L25_IND": st["L25"]["IND"], "L25_EMP": st["L25"]["EMP"], "L25_COM": st["L25"]["COM"]})
            # majorization check (empirical vs comonotone)
            assert D.majorization_check(D.emp_values(Xk, OM), D.com_values(Xk, OM))
    t = pd.DataFrame(cells); t.to_csv(os.path.join(RES, "coupling_cells.csv"), index=False)
    valid = t.dropna(subset=["pos"])
    OUT["coupling"] = {"n_cells": len(t), "n_nondegenerate": int(len(valid)),
                       "pos_median": float(valid.pos.median()), "pos_min": float(valid.pos.min()), "pos_max": float(valid.pos.max()),
                       "n_emp_below_ind": int((valid.sd_EMP < valid.sd_IND - 1e-12).sum()),
                       "emp_le_com_all": bool((t.sd_EMP <= t.sd_COM + 1e-12).all()),
                       "L_order_ok": bool(((t.L10_COM <= t.L10_EMP + 1e-9) & (t.L25_COM <= t.L25_EMP + 1e-9)).all())}
    save_tex("coupling_cells", [f"{r.alt} & $C_{r.crit[1]}$ & {r.S:.4f} & {r.sd_IND:.4f} & {r.sd_EMP:.4f} & {r.sd_COM:.4f} & "
                                f"{('--' if np.isnan(r.pos) else f'{r.pos:.2f}')} & {r.L10_IND:.4f} & {r.L10_EMP:.4f} & {r.L10_COM:.4f} \\\\"
                                for r in t.itertuples()])
    # risk-adjusted rankings
    Sfull = np.array([[stats[(i, j)]["S"] for j in range(5)] for i in range(4)])
    rows = []
    for coup in ["IND", "EMP", "COM"]:
        SD = np.array([[stats[(i, j)]["sd"][coup] for j in range(5)] for i in range(4)])
        for beta in [0, 0.5, 1, 2, 4]:
            val = D.wsm(Sfull - beta * SD, W0)
            rows.append({"score": f"S-{beta}sd", "coupling": coup, **dict(zip(A, val)), "ranking": D.ranking_string(val, A)})
        for al in ["L10", "L25"]:
            Lm = np.array([[stats[(i, j)][al][coup] for j in range(5)] for i in range(4)])
            val = D.mabac(Lm, W0)[0]
            rows.append({"score": al, "coupling": coup, **dict(zip(A, val)), "ranking": D.ranking_string(val, A)})
    r2 = pd.DataFrame(rows); r2.to_csv(os.path.join(RES, "risk_rankings.csv"), index=False)
    OUT["risk_rankings"] = r2.round(5).to_dict(orient="records")
    # reversal thresholds beta*
    Wv = D.wsm(Sfull, W0); thr = []
    for coup in ["IND", "EMP", "COM"]:
        SD = np.array([[stats[(i, j)]["sd"][coup] for j in range(5)] for i in range(4)]); Dv = SD @ W0
        for i in range(4):
            for r_ in range(4):
                if i < r_:
                    dW, dD = Wv[i] - Wv[r_], Dv[i] - Dv[r_]
                    b = dW / dD if abs(dD) > 1e-12 and dW / dD > 0 else np.inf
                    thr.append({"pair": f"{A[i]}-{A[r_]}", "coupling": coup, "dW": dW, "dD": dD, "beta_star": b})
    th = pd.DataFrame(thr); th.to_csv(os.path.join(RES, "reversal_thresholds.csv"), index=False)
    piv = th.pivot(index="pair", columns="coupling", values="beta_star")[["IND", "EMP", "COM"]]
    OUT["beta_star"] = piv.replace(np.inf, None).round(3).to_dict(orient="index")
    save_tex("beta_star", [f"{pr.replace('-', '--')} & " + " & ".join(("$\\infty$" if not np.isfinite(v) else f"{v:.2f}") for v in row) + " \\\\"
                           for pr, row in piv.iterrows()])
    lab = {"S-0sd": "$S$ ($\\beta=0$)", "S-0.5sd": "$S-0.5\\sigma$", "S-1sd": "$S-\\sigma$", "S-2sd": "$S-2\\sigma$",
           "S-4sd": "$S-4\\sigma$", "L10": "$L_{0.10}$ (worst 10\\% of days)", "L25": "$L_{0.25}$"}
    lines = []
    for sc in ["S-0sd", "S-0.5sd", "S-1sd", "S-2sd", "S-4sd", "L10", "L25"]:
        sub = r2[r2.score == sc].set_index("coupling")
        lines.append(f"{lab[sc]} & " + " & ".join(rk(sub.loc[c, 'ranking']) for c in ["IND", "EMP", "COM"]) + " \\\\")
    save_tex("risk_rankings", lines)
    return t, r2


# ============================================================== weights, temporal, sampling
def section_weights(S):
    rows = []
    for nm, w in [("Equal (baseline)", W0), ("ROC, $C_2\\succ C_1\\succ C_4\\succ C_3\\succ C_5$", D.roc_weights([1, 0, 3, 2, 4], 5)),
                  ("ROC, $C_1\\succ C_2\\succ C_3\\succ C_4\\succ C_5$", D.roc_weights([0, 1, 2, 3, 4], 5)),
                  ("Entropy (objective)", D.entropy_weights(S)), ("Stated vector $(0.25,0.25,0.20,0.15,0.15)$", np.array([.25, .25, .2, .15, .15]))]:
        Q = D.mabac(S, w)[0]
        rows.append({"weights": nm, "w": np.round(w, 3).tolist(), **dict(zip(A, Q)), "ranking": D.ranking_string(Q, A)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "criterion_weights.csv"), index=False)
    OUT["criterion_weights"] = t.round(4).to_dict(orient="records")
    save_tex("criterion_weights", [f"{r['weights']} & (" + ", ".join(f"{x:.2f}" for x in r['w']) + ") & " + rk(r["ranking"]) + " \\\\"
                                   for _, r in t.iterrows()])
    b_flat, cw = D.smaa(S, 10_000, None, None, SEED)
    b_conc, _ = D.smaa(S, 10_000, 50.0, W0, SEED)
    OUT["smaa_flat"] = b_flat.round(4).tolist(); OUT["smaa_conc"] = b_conc.round(4).tolist()
    OUT["central_weights"] = np.round(cw, 3).tolist()
    save_tex("smaa", [f"{a} & " + " & ".join(f"{b_flat[i, r]:.3f}" for r in range(4)) + " & " +
                      " & ".join(f"{b_conc[i, r]:.3f}" for r in range(4)) + " & " +
                      ("--" if np.isnan(cw[i]).any() else "(" + ", ".join(f"{x:.2f}" for x in cw[i]) + ")") + " \\\\"
                      for i, a in enumerate(A)])
    return b_flat


def section_temporal(X, R):
    rows = []
    df = R["df"]
    dem = df.groupby("period").load_kw.sum().reindex(P.PERIODS).to_numpy()
    for nm, om in [("Duration $(8,10,6)/24$ (baseline)", OM), ("Equal", np.full(3, 1 / 3)), ("Demand share", dem / dem.sum())] + \
                  [(f"Tariff emphasis $\\lambda={lam:g}$, $\\theta=0.5$", D.temporal_weights_exp(lam, 0.5)) for lam in [0, 0.5, 1, 2]]:
        Q = D.mabac(D.scores_quantile(X, om), W0)[0]
        rows.append({"temporal": nm, "w1": om[0], "w2": om[1], "w3": om[2], **dict(zip(A, Q)), "ranking": D.ranking_string(Q, A)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "temporal_weights.csv"), index=False)
    OUT["temporal"] = t.round(4).to_dict(orient="records")
    save_tex("temporal", [f"{r['temporal']} & {r['w1']:.3f} & {r['w2']:.3f} & {r['w3']:.3f} & " + " & ".join(f4(r[a]) for a in A) +
                          " & " + rk(r["ranking"]) + " \\\\" for _, r in t.iterrows()])


def section_sampling(X, days, S):
    Qs = D.bootstrap_Q(X, OM, W0, B=1000, block=28, seed=SEED)
    np.save(os.path.join(RES, "bootstrap_Q.npy"), Qs)
    Q = D.mabac(S, W0)[0]
    rows = []
    for i in range(4):
        for r_ in range(4):
            if i < r_:
                d = Qs[:, i] - Qs[:, r_]
                rows.append({"pair": f"{A[i]}-{A[r_]}", "point": Q[i] - Q[r_], "lo": np.percentile(d, 2.5),
                             "hi": np.percentile(d, 97.5), "p_pos": float((d > 0).mean())})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "bootstrap_pairs.csv"), index=False)
    OUT["bootstrap_pairs"] = t.round(5).to_dict(orient="records")
    OUT["bootstrap_p_first"] = [float((Qs.argmax(1) == i).mean()) for i in range(4)]
    save_tex("bootstrap", [f"$Q({r['pair'].split('-')[0]})-Q({r['pair'].split('-')[1]})$ & {r['point']:.4f} & "
                           f"[{r['lo']:.4f}, {r['hi']:.4f}] & {r['p_pos']:.3f} \\\\" for _, r in t.iterrows()])
    yrs = pd.DatetimeIndex(days).year; rows = []
    for y in sorted(set(yrs)):
        Qy = D.mabac(D.scores_quantile(X[..., yrs == y], OM), W0)[0]
        rows.append({"year": int(y), **dict(zip(A, Qy)), "ranking": D.ranking_string(Qy, A)})
    ty = pd.DataFrame(rows); ty.to_csv(os.path.join(RES, "per_year.csv"), index=False)
    OUT["per_year"] = ty.round(5).to_dict(orient="records")
    save_tex("per_year", [f"{r['year']} & " + " & ".join(f4(r[a]) for a in A) + " & " + rk(r["ranking"]) + " \\\\" for _, r in ty.iterrows()])
    return Qs, ty


# ============================================================== load models, ceilings, OAT
def section_load_models_and_ceilings(R):
    rows = []
    for lm in ["stochastic", "h0", "metered"]:
        RR = R if lm == "stochastic" else P.run(BASE_PARAMS, load_model=lm)
        Q, _, _, _ = P.base_Q(RR["blocks"])
        rows.append({"case": f"load:{lm}", "targets": f"A2:{RR['targets']['A2']}, A4:{RR['targets']['A4']}",
                     **dict(zip(A, Q)), "ranking": D.ranking_string(Q, A)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "load_model_rankings.csv"), index=False)
    OUT["load_model_rankings"] = t.round(5).to_dict(orient="records")
    nm = {"load:stochastic": "Stochastic neighbourhood (baseline)", "load:h0": "BDEW H0 standard profile",
          "load:metered": "Metered low-voltage aggregate"}
    save_tex("load_model_rankings", [f"{nm[r['case']]} & " + " & ".join(f4(r[a]) for a in A) + " & " + rk(r["ranking"]) + " \\\\"
                                     for _, r in t.iterrows()])
    labs = A + ["L2", "L3", "L4"]
    Q, _, _, _ = P.base_Q(R["blocks"], alts=labs)
    OUT["Q_with_ceilings"] = dict(zip(labs, np.round(Q, 4).tolist()))
    OUT["ranking_with_ceilings"] = D.ranking_string(Q, labs)


def oat_value_label(v):
    return "off" if v == "off" else (f"{v:g}" if isinstance(v, (int, float)) else str(v))


def section_oat():
    specs = [("Battery capacity (kWh)", "battery_kwh", [10.0, 20.0, 30.0, 40.0]),
             ("Grid-import limit (kW)", "grid_max", [10.0, 12.0, 14.0, 16.0, 18.0]),
             ("DR cap $\\rho^{\\max}$", "dr_cap", [0.05, 0.10, 0.15, 0.20]),
             ("DR rebound share $\\beta_{\\mathrm{DR}}$", "dr_rebound", [0.0, 0.5, 1.0]),
             ("Fixed-cost multiplier", "fixed_cost_mult", [0.5, 1.0, 2.0]),
             ("Unserved-energy penalty (USD/kWh)", "c_unmet", [1.0, 5.0, 10.0]),
             ("On-peak price (USD/kWh)", "onpeak_price", [0.20, 0.30, 0.40]),
             ("PV scaling", "pv_scale", [0.8, 1.0, 1.2]),
             ("Wind scaling", "wind_scale", [0.8, 1.0, 1.2])]
    rows = []
    for label, key, vals in specs:
        for v in vals:
            params, pv, wd, prices = {}, 1.0, 1.0, None
            if key == "pv_scale":
                pv = v
            elif key == "wind_scale":
                wd = v
            elif key == "onpeak_price":
                prices = dict(E.CONFIG["prices"]); prices["t3_onpeak"] = v
            else:
                params = {key: v}
            RR = P.run(params, pv_scale=pv, wind_scale=wd, prices=prices)
            Q, _, _, _ = P.base_Q(RR["blocks"])
            rows.append({"parameter": label, "value": v, "targets": f"{RR['targets']['A2']}/{RR['targets']['A4']}",
                         **dict(zip(A, Q)), "ranking": D.ranking_string(Q, A)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "oat.csv"), index=False)
    OUT["oat"] = t.round(5).to_dict(orient="records")
    lines = []
    for par, grp in t.groupby("parameter", sort=False):
        for _, r in grp.iterrows():
            lines.append(f"{par} & {oat_value_label(r['value'])} & {r['targets']} & " + " & ".join(f4(r[a]) for a in A) + " & " + rk(r["ranking"]) + " \\\\")
        lines.append("\\addlinespace")
    save_tex("oat", lines[:-1])
    OUT["oat_first"] = t.ranking.str.split(" > ").str[0].value_counts().to_dict()
    return t


def section_gsa(skip):
    path = os.path.join(RES, "gsa_morris.csv")
    if not (skip and os.path.exists(path)):
        import gsa
        gsa.run_morris(path, seed=SEED)
    g = pd.read_csv(path)
    import gsa
    idx = gsa.indices(g)
    idx.to_csv(os.path.join(RES, "gsa_indices.csv"), index=False)
    OUT["gsa_winner_share"] = g["winner"].value_counts(normalize=True).round(4).to_dict()
    OUT["gsa_n"] = int(len(g))
    OUT["gsa_indices"] = idx.round(5).to_dict(orient="records")
    names = {"grid_max": "Grid-import limit (kW)", "c_unmet": "Unserved-energy penalty (USD/kWh)",
             "battery_kwh": "Battery capacity (kWh)", "dr_cap": "DR cap $\\rho^{\\max}$", "tariff_ratio": "On-/off-peak price ratio"}
    lines = []
    for f_ in gsa.FACTORS:
        sub = idx[idx.factor == f_[0]].set_index("output")
        lines.append(f"{names[f_[0]]} & [{f_[1]:g}, {f_[2]:g}] & " +
                     " & ".join(f"{sub.loc[o, 'mu_star']:.4f} & {sub.loc[o, 'sigma']:.4f}" for o in gsa.OUTPUTS) + " \\\\")
    save_tex("gsa", lines)
    return g, idx


def main():
    t0 = time.time()
    skip_gsa = "--skip-gsa" in sys.argv
    if "--figures-only" in sys.argv:
        import figures
        figures.from_results(); return
    section_data()
    R = section_baseline()
    Q, S, X, days, H, Dv = section_scores(R)
    section_context(H)
    section_criteria(R)
    section_hesitancy(X, S)
    section_operators(X, H)
    section_normalization(R, S)
    cells, risk = section_coupling(X, S)
    b_flat = section_weights(S)
    section_temporal(X, R)
    Qs, per_year = section_sampling(X, days, S)
    section_load_models_and_ceilings(R)
    oat = section_oat()
    gsa_res, gsa_idx = section_gsa(skip_gsa)
    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(OUT, fh, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else o.item())
    import figures
    figures.from_results()
    print(json.dumps({k: OUT[k] for k in ("ranking", "Q", "targets", "bootstrap_p_first")}, default=str))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
