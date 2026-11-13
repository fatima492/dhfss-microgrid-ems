"""Reproduce every table and figure of the revised manuscript and its supplement.

    python src/gen_household_load.py   # only if data/household_load_15min.csv is missing (~15 min)
    python src/build_sydney_data.py "<Ausgrid 2012-2013 csv>"   # only if data/sydney_ausgrid_15min.csv is missing
    python run_all.py                  # full run (about 45-60 min on 2 cores)
    python run_all.py --skip-gsa       # reuse results/gsa_morris.csv
    python run_all.py --figures-only   # redraw figures from results/

Outputs: results/*.csv, results/summary.json, results/tables/*.tex, figures/*.png

Revision (November 2026), in response to the reviewers:
* six candidate policies: A2/A4 tuned to operating cost and A5/A6 tuned to the multi-criteria
  objective over the same rule space (grid-charging target x discharge mode)          [R2.4, R3.3]
* sixth criterion C6 (demand-side comfort) and the end-use service ratio as a C2 variant [R2.3, R3.2]
* limited-foresight controllers L2F/L4F next to the perfect-foresight LPs              [R2.5]
* comparison with conventional crisp MCDM methods                                      [R3.1]
* joint sensitivity analysis and preference regions                                     [R3.5]
* second site with measured household load and PV (Sydney, Ausgrid)                    [R3.4]
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
             "pvlib": "pvlib", "demandlib": "demandlib", "SALib": "SALib", "sklearn": "scikit-learn"}
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
A = P.ALTS                                        # A1 ... A6
CR = M.CRITERIA                                   # C1 ... C6
NA, NC = len(A), len(CR)
BENCH = ["L2", "L3", "L4", "L2F", "L4F"]
BASE_PARAMS = {}                                  # M.BASE already holds the baseline (grid limit 14 kW)
OM = P.OMEGA_DURATION
W0 = P.W_EQUAL
OUT = {}


def lab_tex(x):
    return x[0] + "_{" + x[1:] + "}"


def rk(s):
    parts = [x.strip() for x in str(s).split(">") if x.strip()]
    if not parts or str(s) == "nan":
        return "none feasible"
    return "$" + "{\\succ}".join(lab_tex(x) for x in parts) + "$"


def f4(x):
    return f"{x:.4f}"


def save_tex(name, lines):
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def mabac_rank(S, w=None, labels=A):
    w = W0 if w is None else w
    Q = D.mabac(S, w)[0]
    return Q, D.ranking_string(Q, labels)


def pairs(labels):
    return [(i, r) for i in range(len(labels)) for r in range(len(labels)) if i < r]


# ============================================================== data and baseline
def section_data():
    df = P.inputs("stochastic")
    diag = E.diagnostics(df[df.index.year.isin(E.TEST_YEARS)])
    rows = {}
    cases = [("stochastic", "zurich"), ("h0", "zurich"), ("metered", "zurich"), ("measured", "sydney")]
    for lm, site in cases:
        if site == "sydney":
            d = P.inputs(site="sydney")                  # full year (the test split has gaps)
        else:
            d = P.inputs(lm); d = d[d.index.year.isin(E.TEST_YEARS)]
        g = E.diagnostics(d)
        rows[lm] = {"peak_hour": int(g["hourly"]["load_kw"].idxmax()),
                    "peak_to_mean": float(g["hourly"]["load_kw"].max() / g["hourly"]["load_kw"].mean()),
                    "daily_cv": g["daily_cv"], "daily_resid_cv": g["daily_resid_cv"],
                    "acf1": g["acf"]["load_kw"][1], "acf96": g["acf"]["load_kw"][96],
                    "max_kw": float(d.load_kw.max()), "pv_kwh_per_kwp_yr": float(d.pv_kw.mean() * 8760 / E.CONFIG["pv_kwp"]),
                    **{f"mean_{k}": float(v) for k, v in g["period_means"]["load_kw"].items()}}
    lm_tab = pd.DataFrame(rows).T; lm_tab.to_csv(os.path.join(RES, "load_models.csv"))
    OUT["load_models"] = lm_tab.round(4).to_dict(orient="index")
    full = E.diagnostics(df)
    full["annual"].to_csv(os.path.join(RES, "annual_inputs.csv"))
    OUT["annual_inputs"] = full["annual"].round(1).to_dict(orient="index")
    OUT["period_means_test"] = diag["period_means"].round(3).to_dict()
    OUT["acf_test"] = diag["acf"]
    OUT["pv_night_share"] = full["pv_night_share"]
    names = {"stochastic": "Zurich, stochastic neighbourhood (baseline)", "h0": "Zurich, BDEW H0 standard profile",
             "metered": "Zurich, metered low-voltage aggregate", "measured": "Sydney, measured households (Ausgrid)"}
    save_tex("load_models", [
        f"{names[k]} & {int(r['peak_hour']):d}:00 & {r['peak_to_mean']:.2f} & {r['mean_t1_offpeak']:.2f} & "
        f"{r['mean_t2_midpeak']:.2f} & {r['mean_t3_onpeak']:.2f} & {100*r['daily_resid_cv']:.1f} & {r['acf1']:.3f} \\\\"
        for k, r in lm_tab.iterrows()])
    a = full["annual"]
    save_tex("annual_inputs", [f"{y} & {r.ghi_kwh_m2:,.0f} & {r.pv_kwh:,.0f} & {r.wind_kwh:,.0f} & {r.load_kwh:,.0f} \\\\"
                               for y, r in a.iterrows()])
    # tariff reference price of the DR rule (R1.2): duration-weighted daily mean of the tariff
    pr = E.CONFIG["prices"]
    OUT["pibar_duration_weighted"] = float(sum(pr[k] * h for k, h in zip(E.PERIODS, E.PERIOD_HOURS)) / E.PERIOD_HOURS.sum())
    OUT["pibar_arithmetic_levels"] = float(np.mean(list(pr.values())))
    OUT["pibar_in_code"] = float(df[df.index.year.isin(E.TEST_YEARS)].price.mean())
    return df


def kpi_rows(sims, p, ny):
    rows = []
    for a, s in sims.items():
        forgone = (1 - p["dr_rebound"]) * s.dr.sum()
        rows.append({"policy": a, "cost_usd": s.cost.sum() / ny, "energy_cost_usd": s.energy_cost.sum() / ny,
                     "unmet_kwh": s.unmet.sum() * M.DT / ny, "grid_kwh": s.grid.sum() * M.DT / ny,
                     "grid_charge_kwh": s.ch_g.sum() * M.DT / ny, "discharge_kwh": s.dis.sum() * M.DT / ny,
                     "discharge_re_kwh": s.dis_re.sum() * M.DT / ny, "curt_kwh": s.curt.sum() * M.DT / ny,
                     "dr_kwh": s.dr.sum() * M.DT / ny, "forgone_kwh": forgone * M.DT / ny,
                     "reliability": 1 - s.unmet.sum() / s.adj.sum(),
                     "service_ratio": 1 - (s.unmet.sum() + forgone) / s.req.sum(),
                     "self_consumption": (s.re_load.sum() + s.ch_re.sum()) / s.re.sum(),
                     "cost_per_served": s.energy_cost.sum() / ((s.adj - s.unmet).sum() * M.DT)})
    return pd.DataFrame(rows).set_index("policy")


def section_baseline():
    R = P.run(BASE_PARAMS, with_lp=True, with_forecast=True)
    OUT["targets"] = {a: P.setting_label(s) for a, s in R["targets"].items()}
    tt = R["tune_table"]; tt.to_csv(os.path.join(RES, "tuning_train.csv"), index=False)
    ny_tr = len(E.TRAIN_YEARS)
    lines = []
    for st in P.SETTINGS:
        cells = []
        for hw, ac, am in (("A2", "A2", "A5"), ("A4", "A4", "A6")):
            r = tt[(tt.hardware == hw) & (tt["mode"] == st[0]) & (tt.target.astype(str) == str(st[1]))].iloc[0]
            c = f"{r.train_cost_usd/ny_tr:,.0f}"; w = f"{r.train_W:.4f}"
            if R["targets"][ac] == st:
                c = "\\textbf{" + c + "}"
            if R["targets"][am] == st:
                w = "\\textbf{" + w + "}"
            cells += [c, w]
        lines.append(P.setting_label(st).replace("%", "\\%") + " & " + " & ".join(cells) + " \\\\")
    save_tex("tuning", lines)
    ny = len(E.TEST_YEARS)
    kpi = kpi_rows(R["sims"], R["p"], ny); kpi.to_csv(os.path.join(RES, "kpi_test.csv"))
    OUT["kpi"] = kpi.round(4).to_dict(orient="index")
    lines = []
    for a, r in kpi.iterrows():
        if a == "L2":
            lines.append("\\midrule")
        reo = (f"{100*r.discharge_re_kwh/r.discharge_kwh:.0f}" if r.discharge_kwh > 0 else "--")
        lines.append(f"${lab_tex(a)}$ {M.SHORT[a]} & {r.energy_cost_usd:,.0f} & {100*r.cost_per_served:.2f} & {r.unmet_kwh:,.0f} & "
                     f"{100*r.reliability:.2f} & {r.dr_kwh:,.0f} & {100*r.service_ratio:.2f} & {r.grid_charge_kwh:,.0f} & "
                     f"{r.discharge_kwh:,.0f} & {reo} \\\\")
    save_tex("kpi", lines)
    return R


def section_scores(R):
    Q, S, X, days = P.base_Q(R["blocks"])
    OUT["n_days"] = len(days); OUT["S"] = S.round(4).tolist(); OUT["Q"] = Q.round(4).tolist()
    OUT["ranking"] = D.ranking_string(Q, A)
    Qv, V, g, Qm = D.mabac(S, W0)
    rnk = D.ranks(Q)
    H = D.build_hfes(X, "quantile")
    pd.DataFrame(S, index=A, columns=CR).to_csv(os.path.join(RES, "scores.csv"))
    pd.DataFrame(Qm, index=A, columns=CR).to_csv(os.path.join(RES, "mabac_distances.csv"))
    save_tex("mabac", [f"${lab_tex(a)}$ & " + " & ".join(f"{S[i, j]:.4f}" for j in range(NC)) + f" & {Q[i]:.4f} & {rnk[i]} \\\\"
                       for i, a in enumerate(A)] + ["\\midrule", "$g_j$ (on $v_{ij}$) & " + " & ".join(f"{x:.4f}" for x in g) + " & & \\\\"])
    for k, per in enumerate(P.PERIODS):
        save_tex(f"hfe_{per}", [f"${lab_tex(a)}$ & " + " & ".join("\\{" + ",".join(f"{v:.3f}" for v in H[i][j][k]) + "\\}"
                                                              for j in range(NC)) + " \\\\" for i, a in enumerate(A)])
    pooled = pd.concat([R["blocks"][a][CR] for a in A]); corr = pooled.corr()
    corr.to_csv(os.path.join(RES, "criterion_correlation.csv"))
    save_tex("criterion_corr", [f"$C_{c[1]}$ & " + " & ".join(f"{corr.loc[c, d]:.2f}" for d in CR) + " \\\\" for c in CR])
    OUT["max_offdiag_corr"] = float(np.nanmax(np.abs(corr.values - np.eye(NC))))
    OUT["n_hfes"] = NA * NC * 3
    OUT["degenerate_hfes"] = int(sum(np.ptp(H[i][j][k]) < 1e-9 for i in range(NA) for j in range(NC) for k in range(3)))
    OUT["hfes_with_one"] = int(sum(np.isclose(H[i][j][k], 1).any() for i in range(NA) for j in range(NC) for k in range(3)))
    np.save(os.path.join(RES, "block_tensor_test.npy"), X)
    pd.Series([str(d.date()) for d in days]).to_csv(os.path.join(RES, "block_days.csv"), index=False, header=["opday"])
    return Q, S, X, days, H


# ============================================================== context-dependent parameter sets (DHFSS)
def section_context(H):
    """C3 inactive off-peak and on-peak (A_t1 = A_t3 = C minus C3); temporal weights renormalized."""
    p = 3
    Sk = np.array([[[np.mean(H[i][j][k]) for k in range(p)] for j in range(NC)] for i in range(NA)])
    rows, lines = [], []
    for label, inactive in (("All criteria active in every period ($A_t=\\mathcal{C}$, baseline)", []),
                            ("$C_3$ inactive off-peak and on-peak ($A_{t_1}=A_{t_3}=\\mathcal{C}\\setminus\\{C_3\\}$)", [(2, 0), (2, 2)])):
        act = np.ones((NC, p), bool)
        for j, k in inactive:
            act[j, k] = False
        S = np.zeros((NA, NC))
        for j in range(NC):
            w = np.asarray(OM) * act[j]; S[:, j] = Sk[:, j, :] @ (w / w.sum())
        Q = D.mabac(S, W0)[0]
        rows.append({"variant": label, **{a: Q[i] for i, a in enumerate(A)}, "C3_scores": S[:, 2].round(4).tolist(),
                     "ranking": D.ranking_string(Q, A)})
        lines.append(f"{label} & " + " & ".join(f4(q) for q in Q) + " & " + rk(D.ranking_string(Q, A)) + " \\\\")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "context_parameters.csv"), index=False)
    OUT["context_dhfss"] = [{k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in r.items()} for r in rows]
    save_tex("context", lines)


# ============================================================== criterion definitions, comfort and reliability
def section_criteria(R):
    p = R["p"]
    rows = []
    variants = [
        ("Baseline: $C_1$--$C_6$ ($C_2$ supply reliability, $C_6$ comfort with $\\delta=0.5$)", CR),
        ("First-submission specification: $C_1$--$C_5$ (no comfort criterion)", M.CRITERIA_V1),
        ("$C_2$ end-use service ratio (counts forgone curtailment), no $C_6$", ["C1", "C2_svc", "C3", "C4", "C5"]),
        ("$C_2$ end-use service ratio and $C_6$", ["C1", "C2_svc", "C3", "C4", "C5", "C6"]),
        ("$C_6$ with $\\delta=0$ (forgone energy only)", ["C1", "C2", "C3", "C4", "C5", "C6_forgone"]),
        ("$C_6$ with $\\delta=1$ (all curtailed energy)", ["C1", "C2", "C3", "C4", "C5", "C6_all"]),
        ("$C_1$ total cost, no penalty", ["C1_tot", "C2", "C3", "C4", "C5", "C6"]),
        ("$C_1$ total cost incl. unserved-energy penalty", ["C1_pen", "C2", "C3", "C4", "C5", "C6"]),
        ("$C_1$ linear-clipped transform $1-y/\\rho$", ["C1_lin", "C2", "C3", "C4", "C5", "C6"]),
        ("$C_4$ local supply incl. grid-charged storage", ["C1", "C2", "C3", "C4_local", "C5", "C6"]),
        ("$C_1$ with penalty and $C_4$ local (both comparators)", ["C1_pen", "C2", "C3", "C4_local", "C5", "C6"])]
    for nm, cr in variants:
        Q, _, _, _ = P.base_Q(R["blocks"], criteria=cr)
        rows.append({"variant": nm, **{a: Q[i] for i, a in enumerate(A)}, "ranking": D.ranking_string(Q, A)})
    n_def = len(rows)
    _, S, _, _ = P.base_Q(R["blocks"])
    for j in range(NC):
        w = W0.copy(); w[j] = 0; w /= w.sum(); Q = D.mabac(S, w)[0]
        rows.append({"variant": f"leave out $C_{j+1}$", **{a: Q[i] for i, a in enumerate(A)}, "ranking": D.ranking_string(Q, A)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "criterion_variants.csv"), index=False)
    OUT["criterion_variants"] = t.round(4).to_dict(orient="records")
    save_tex("criterion_variants", [f"{r['variant']} & " + " & ".join(f4(r[a]) for a in A) + " & " + rk(r["ranking"]) + " \\\\"
                                    + ("\n\\midrule" if i == n_def - 1 else "") for i, r in t.iterrows()])
    # reliability (and end-use service) as a constraint
    sims = R["sims"]
    rel = {a: 1 - sims[a].unmet.sum() / sims[a].adj.sum() for a in A}
    svc = {a: 1 - (sims[a].unmet.sum() + (1 - p["dr_rebound"]) * sims[a].dr.sum()) / sims[a].req.sum() for a in A}
    OUT["reliability_test"] = rel; OUT["service_ratio_test"] = svc
    others = [j for j in range(NC) if CR[j] != "C2"]
    rows = []
    for measure, vals, thrs in (("supply reliability", rel, [0.985, 0.990, 0.995, 0.9975]),
                                ("end-use service ratio", svc, [0.95, 0.97, 0.99])):
        for thr in thrs:
            keep = [i for i, a in enumerate(A) if vals[a] >= thr]
            if len(keep) >= 2:
                S2 = S[keep][:, others]; w = np.full(len(others), 1.0 / len(others))
                Q = D.mabac(S2, w)[0]; rnk = D.ranking_string(Q, [A[i] for i in keep])
            else:
                rnk = " > ".join(A[i] for i in keep)
            rows.append({"measure": measure, "threshold": thr, "feasible": ", ".join(A[i] for i in keep), "ranking": rnk})
    t2 = pd.DataFrame(rows); t2.to_csv(os.path.join(RES, "reliability_constraint.csv"), index=False)
    OUT["reliability_constraint"] = t2.to_dict(orient="records")
    save_tex("reliability_constraint", [f"{r['measure'].capitalize()} $\\geq$ {100*r['threshold']:.2f}\\% & "
                                        + (", ".join(f"${lab_tex(x.strip())}$" for x in r["feasible"].split(",")) if r["feasible"] else "--")
                                        + f" & {rk(r['ranking'])} \\\\" for _, r in t2.iterrows()])


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
                             "ranking": D.ranking_string(Q, A)})
    Hagg, Sx, _ = D.aggregate_matrix(H, OM, "IND", "AR")
    for meth, vals in [("WSM", D.wsm(Sx, W0)), ("TOPSIS", D.topsis(Sx, W0)[0]), ("HF-TOPSIS", D.hf_topsis(Hagg, W0)[0])]:
        rows.append({"operator": "IND-AR", "eps": 0.0, "method": meth, **dict(zip(A, vals)), "ranking": D.ranking_string(vals, A)})
    t = pd.DataFrame(rows)

    def guaranteed(r):                             # R2.6: which agreements follow from the theory
        if r["operator"] == "IND-AR" and r["method"] == "MABAC":
            return "baseline"
        if r["operator"] == "COM-AR" and r["method"] == "MABAC":
            return "yes (Thm.~3(i): same scores)"
        if r["operator"] == "IND-AR" and r["method"] == "WSM":
            return "yes (Prop.~5: same order)"
        return "no"
    t["guaranteed"] = [guaranteed(r) for _, r in t.iterrows()]
    base_rank = t.iloc[0]["ranking"]
    t["same_as_baseline"] = t["ranking"] == base_rank
    t.to_csv(os.path.join(RES, "operators_methods.csv"), index=False)
    OUT["operators_methods"] = t.round(5).to_dict(orient="records")
    OUT["operators_summary"] = {"n": int(len(t)), "n_same": int(t.same_as_baseline.sum()),
                                "n_guaranteed": int((t.guaranteed.str.startswith("yes")).sum()),
                                "n_informative": int((t.guaranteed == "no").sum()),
                                "n_informative_same": int(((t.guaranteed == "no") & t.same_as_baseline).sum())}

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
        lines.append(f"{label(r)} & " + " & ".join(str(x) for x in rr) + " & " + rk(r["ranking"]) + f" & {r['guaranteed']} \\\\")
    save_tex("operators_ranks", lines)
    sel = t[(t.eps == 0.0)]
    save_tex("operators_scores", [f"{label(r)} & " + " & ".join(f4(r[a]) for a in A) + " \\\\" for _, r in sel.iterrows()])
    OUT["absorbing_by_criterion"] = {CR[j]: int(sum(np.isclose(H[i][j][k], 1).any() for i in range(NA) for k in range(3)))
                                     for j in range(NC)}
    _, Se, _ = D.aggregate_matrix(H, OM, "IND", "EIN", 0.0)
    OUT["absorbed_cells_EIN"] = int(np.isclose(Se, 1.0).sum())
    OUT["absorbed_by_criterion_EIN"] = {CR[j]: int(np.isclose(Se[:, j], 1.0).sum()) for j in range(NC)}


def _period_sums(blocks):
    g = blocks.groupby("period")
    keys = ["energy_cost", "served", "unmet", "adj", "re", "re_load", "ch_re", "dis_re", "dr", "req"]
    out = pd.DataFrame({k: g[k].sum() for k in keys}); out["mean_block_peak_kw"] = g["peak"].mean()
    return out.reindex(P.PERIODS)


def raw_period_totals(blocks, p):
    a_c = (1 - p["dr_rebound"]) + p["comfort_delta"] * p["dr_rebound"]
    def raw(t):
        return np.stack([t.energy_cost / t.served, 1 - t.unmet / t.adj, (t.re_load + t.ch_re) / t.re,
                         (t.re_load + t.dis_re) / t.adj, t.mean_block_peak_kw, 1 - a_c * t.dr / t.req], axis=0)
    return np.stack([raw(_period_sums(blocks[a])) for a in A])


def section_normalization(R, S):
    p = R["p"]; rows = []
    COST = np.array([j in (0, 4) for j in range(NC)])
    Q, r = mabac_rank(S); rows.append({"scheme": "N1", **dict(zip(A, Q)), "ranking": r})
    Rr = raw_period_totals(R["blocks"], p)
    df = R["df"]; gmax = p["grid_max"]
    imp = np.minimum(df.load_kw, gmax)
    ref = (df.assign(c=df.price * imp, s=imp).groupby("period")[["c", "s"]].sum()).reindex(P.PERIODS)
    rho = (ref.c / ref.s).to_numpy()
    Z = Rr.copy(); Z[:, 0, :] = rho / (rho + Rr[:, 0, :]); Z[:, 4, :] = 1 - Rr[:, 4, :] / gmax
    Q, r = mabac_rank(np.tensordot(Z, OM, axes=([2], [0]))); rows.append({"scheme": "N1c", **dict(zip(A, Q)), "ranking": r})
    Z = np.zeros_like(Rr)
    for j in range(NC):
        for k in range(3):
            v = Rr[:, j, k]; lo, hi = v.min(), v.max()
            Z[:, j, k] = 0.5 if np.isclose(lo, hi) else ((hi - v) / (hi - lo) if COST[j] else (v - lo) / (hi - lo))
    Q, r = mabac_rank(np.tensordot(Z, OM, axes=([2], [0]))); rows.append({"scheme": "N2", **dict(zip(A, Q)), "ranking": r})
    Z = np.zeros_like(Rr)
    for j in range(NC):
        for k in range(3):
            v = Rr[:, j, k]
            Z[:, j, k] = (v.min() / v) if COST[j] else (v / v.max() if v.max() > 0 else 1.0)
    Q, r = mabac_rank(np.tensordot(Z, OM, axes=([2], [0]))); rows.append({"scheme": "N3", **dict(zip(A, Q)), "ranking": r})
    Q = interval_minmax(R); rows.append({"scheme": "N4", **dict(zip(A, Q)), "ranking": D.ranking_string(Q, A)})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "normalization.csv"), index=False)
    OUT["normalization"] = t.round(5).to_dict(orient="records")
    names = {"N1": "Reference-point, daily-block quantile HFEs (baseline)",
             "N1c": "Reference-point, period totals (crisp)",
             "N2": "Min--max across alternatives, period totals (crisp)",
             "N3": "Linear ratio to best, period totals (crisp)",
             "N4": "Interval-level min--max, ties set to 1, $\\{\\mu\\pm\\sigma,\\mu\\}$ (common practice)"}
    save_tex("normalization", [f"{names[r['scheme']]} & " + " & ".join(f4(r[a]) for a in A) + " & " + rk(r["ranking"]) + " \\\\"
                               for _, r in t.iterrows()])


def interval_minmax(R):
    """Common practice: min-max across alternatives at every 15-min step, ties = 1, mean +- sd HFEs."""
    p = R["p"]; gmax = p["grid_max"]; per = R["df"]["period"].to_numpy()
    a_c = (1 - p["dr_rebound"]) + p["comfort_delta"] * p["dr_rebound"]
    C = []
    for a in A:
        s = R["sims"][a]
        with np.errstate(divide="ignore", invalid="ignore"):
            C.append(np.stack([s.energy_cost.to_numpy(), 1 - s.unmet / s.adj, np.where(s.re > 1e-9, (s.re_load + s.ch_re) / s.re, np.nan),
                               (s.re_load + s.dis_re) / s.adj, 1 - s.grid / gmax, 1 - a_c * s.dr / s.req]))
    C = np.stack(C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo = np.nanmin(C, axis=0); hi = np.nanmax(C, axis=0)
    eq = np.isclose(lo, hi)
    span = np.where(eq, 1.0, hi - lo)
    Z = np.where(np.arange(NC)[None, :, None] == 0, (hi - C) / span, (C - lo) / span)
    Z = np.where(eq, 1.0, Z); Z = np.where(np.isnan(C), np.nan, Z)
    S = np.zeros((NA, NC))
    for k, pk in enumerate(P.PERIODS):
        msk = per == pk
        for i in range(NA):
            for j in range(NC):
                z = Z[i, j, msk]; z = z[~np.isnan(z)]
                mu, sd = z.mean(), z.std()
                h = np.unique(np.round(np.clip([mu - sd, mu, mu + sd], 0, 1), 6))
                S[i, j] += OM[k] * h.mean()
    return D.mabac(S, W0)[0]


# ============================================================== coupling and risk
def coupling_cells(X, omega=OM):
    stats = {}
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            stats[(i, j)] = D.coupling_stats(X[i, j], omega, alphas=(0.10, 0.25))
    return stats


def section_coupling(X, S):
    cells, stats = [], coupling_cells(X)
    for i, a in enumerate(A):
        for j, c in enumerate(CR):
            st = stats[(i, j)]
            sdI, sdE, sdC = st["sd"]["IND"], st["sd"]["EMP"], st["sd"]["COM"]
            pos = (sdE - sdI) / (sdC - sdI) if sdC - sdI > 1e-12 else np.nan
            Xk = X[i, j]; cov = np.cov(Xk, bias=True)
            wc = sum(OM[k] * OM[l] * cov[k, l] for k in range(3) for l in range(3) if k < l)
            cells.append({"alt": a, "crit": c, "S": st["S"], "sd_IND": sdI, "sd_EMP": sdE, "sd_COM": sdC,
                          "sd_upper": st["sd_bound_upper"], "pos": pos, "w_cov_sum": wc,
                          "L10_IND": st["L10"]["IND"], "L10_EMP": st["L10"]["EMP"], "L10_COM": st["L10"]["COM"],
                          "L25_IND": st["L25"]["IND"], "L25_EMP": st["L25"]["EMP"], "L25_COM": st["L25"]["COM"]})
            assert D.majorization_check(D.emp_values(Xk, OM), D.com_values(Xk, OM))
    t = pd.DataFrame(cells); t.to_csv(os.path.join(RES, "coupling_cells.csv"), index=False)
    valid = t.dropna(subset=["pos"])
    OUT["coupling"] = {"n_cells": len(t), "n_nondegenerate": int(len(valid)),
                       "pos_median": float(valid.pos.median()), "pos_min": float(valid.pos.min()), "pos_max": float(valid.pos.max()),
                       "n_emp_below_ind": int((valid.sd_EMP < valid.sd_IND - 1e-12).sum()),
                       "cells_emp_below_ind": [f"{r.alt}-{r.crit}" for r in valid.itertuples() if r.sd_EMP < r.sd_IND - 1e-12],
                       "emp_le_com_all": bool((t.sd_EMP <= t.sd_COM + 1e-12).all()),
                       "L_order_ok": bool(((t.L10_COM <= t.L10_EMP + 1e-9) & (t.L25_COM <= t.L25_EMP + 1e-9)).all())}
    save_tex("coupling_cells", [f"${lab_tex(r.alt)}$ & $C_{r.crit[1]}$ & {r.S:.4f} & {r.sd_IND:.4f} & {r.sd_EMP:.4f} & {r.sd_COM:.4f} & "
                                f"{('--' if np.isnan(r.pos) else f'{r.pos:.2f}')} & {r.L10_IND:.4f} & {r.L10_EMP:.4f} & {r.L10_COM:.4f} \\\\"
                                for r in t.itertuples()])
    Sfull = np.array([[stats[(i, j)]["S"] for j in range(NC)] for i in range(NA)])
    rows = []
    for coup in ["IND", "EMP", "COM"]:
        SD = np.array([[stats[(i, j)]["sd"][coup] for j in range(NC)] for i in range(NA)])
        for beta in [0, 0.5, 1, 2, 4]:
            val = D.wsm(Sfull - beta * SD, W0)
            rows.append({"score": f"S-{beta}sd", "coupling": coup, **dict(zip(A, val)), "ranking": D.ranking_string(val, A)})
        for al in ["L10", "L25"]:
            Lm = np.array([[stats[(i, j)][al][coup] for j in range(NC)] for i in range(NA)])
            val = D.mabac(Lm, W0)[0]
            rows.append({"score": al, "coupling": coup, **dict(zip(A, val)), "ranking": D.ranking_string(val, A)})
    r2 = pd.DataFrame(rows); r2.to_csv(os.path.join(RES, "risk_rankings.csv"), index=False)
    OUT["risk_rankings"] = r2.round(5).to_dict(orient="records")
    Wv = D.wsm(Sfull, W0); thr = []
    for coup in ["IND", "EMP", "COM"]:
        SD = np.array([[stats[(i, j)]["sd"][coup] for j in range(NC)] for i in range(NA)]); Dv = SD @ W0
        for i, r_ in pairs(A):
            a_, b_ = (i, r_) if Wv[i] >= Wv[r_] else (r_, i)       # first-named has the higher mean score
            dW, dD = Wv[a_] - Wv[b_], Dv[a_] - Dv[b_]
            b = dW / dD if dD > 1e-12 else np.inf
            thr.append({"pair": f"{A[a_]}-{A[b_]}", "coupling": coup, "dW": dW, "dD": dD, "beta_star": b})
    th = pd.DataFrame(thr); th.to_csv(os.path.join(RES, "reversal_thresholds.csv"), index=False)
    piv = th.pivot(index="pair", columns="coupling", values="beta_star")[["IND", "EMP", "COM"]]
    order = sorted(piv.index, key=lambda s: (A.index(s.split("-")[0]), A.index(s.split("-")[1])))
    piv = piv.loc[order]
    OUT["beta_star"] = piv.replace(np.inf, None).round(3).to_dict(orient="index")
    # pair shown in the coupling figure: the pair whose decision depends most on the coupling within beta <= 4
    cand = []
    for pr, row in piv.iterrows():
        v = row.to_numpy(float); fin = v[np.isfinite(v)]
        if len(fin) and fin.min() <= 4:
            spread = np.ptp(np.where(np.isfinite(v), np.minimum(v, 4), 4))
            cand.append((spread, pr))
    OUT["coupling_pair"] = max(cand)[1] if cand else piv.index[0]
    save_tex("beta_star", [f"${lab_tex(pr.split('-')[0])}$--${lab_tex(pr.split('-')[1])}$ & " +
                           " & ".join(("$\\infty$" if not np.isfinite(v) else f"{v:.2f}") for v in row) + " \\\\"
                           for pr, row in piv.iterrows()])
    lab = {"S-0sd": "$S$ ($\\beta=0$)", "S-0.5sd": "$S-0.5\\sigma$", "S-1sd": "$S-\\sigma$", "S-2sd": "$S-2\\sigma$",
           "S-4sd": "$S-4\\sigma$", "L10": "$L_{0.10}$ (worst 10\\% of days)", "L25": "$L_{0.25}$"}
    lines = []
    for sc in ["S-0sd", "S-0.5sd", "S-1sd", "S-2sd", "S-4sd", "L10", "L25"]:
        sub = r2[r2.score == sc].set_index("coupling")
        lines.append(f"{lab[sc]} & " + " & ".join(rk(sub.loc[c, 'ranking']) for c in ["IND", "EMP", "COM"]) + " \\\\")
    save_tex("risk_rankings", lines)
    return t, r2, stats


# ============================================================== weights, temporal, sampling
def section_weights(S):
    rows = []
    for nm, w in [("Equal (baseline)", W0),
                  ("ROC, $C_2\\succ C_1\\succ C_6\\succ C_4\\succ C_3\\succ C_5$", D.roc_weights([1, 0, 5, 3, 2, 4], NC)),
                  ("ROC, $C_1\\succ C_2\\succ C_3\\succ C_4\\succ C_5\\succ C_6$", D.roc_weights([0, 1, 2, 3, 4, 5], NC)),
                  ("ROC, $C_4\\succ C_3\\succ C_1\\succ C_2\\succ C_5\\succ C_6$", D.roc_weights([3, 2, 0, 1, 4, 5], NC)),
                  ("Entropy (objective)", D.entropy_weights(S)),
                  ("Stated vector", np.array([.20, .20, .15, .15, .15, .15]))]:
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
    save_tex("smaa", [f"${lab_tex(a)}$ & " + " & ".join(f"{b_flat[i, r]:.3f}" for r in range(NA)) + f" & {b_conc[i, 0]:.3f} & " +
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
    for i, r_ in pairs(A):
        d = Qs[:, i] - Qs[:, r_]
        rows.append({"pair": f"{A[i]}-{A[r_]}", "point": Q[i] - Q[r_], "lo": np.percentile(d, 2.5),
                     "hi": np.percentile(d, 97.5), "p_pos": float((d > 0).mean())})
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "bootstrap_pairs.csv"), index=False)
    OUT["bootstrap_pairs"] = t.round(5).to_dict(orient="records")
    OUT["bootstrap_p_first"] = [float((Qs.argmax(1) == i).mean()) for i in range(NA)]
    save_tex("bootstrap", [f"$Q({lab_tex(r['pair'].split('-')[0])})-Q({lab_tex(r['pair'].split('-')[1])})$ & {r['point']:.4f} & "
                           f"[{r['lo']:.4f}, {r['hi']:.4f}] & {r['p_pos']:.3f} \\\\" for _, r in t.iterrows()])
    yrs = pd.DatetimeIndex(days).year; rows = []
    for y in sorted(set(yrs)):
        Qy = D.mabac(D.scores_quantile(X[..., yrs == y], OM), W0)[0]
        rows.append({"year": int(y), **dict(zip(A, Qy)), "ranking": D.ranking_string(Qy, A)})
    ty = pd.DataFrame(rows); ty.to_csv(os.path.join(RES, "per_year.csv"), index=False)
    OUT["per_year"] = ty.round(5).to_dict(orient="records")
    save_tex("per_year", [f"{r['year']} & " + " & ".join(f4(r[a]) for a in A) + " & " + rk(r["ranking"]) + " \\\\" for _, r in ty.iterrows()])
    return Qs, ty


# ============================================================== load models, second site, benchmarks
def section_load_models_and_benchmarks(R):
    rows = []
    for lm, site in [("stochastic", "zurich"), ("h0", "zurich"), ("metered", "zurich"), ("measured", "sydney")]:
        if lm == "stochastic":
            RR = R
        elif site == "sydney":
            RR = P.run(BASE_PARAMS, site="sydney", with_forecast=False)
        else:
            RR = P.run(BASE_PARAMS, load_model=lm)
        Q, S_, _, days_ = P.base_Q(RR["blocks"])
        Q5, _, _, _ = P.base_Q(RR["blocks"], criteria=M.CRITERIA_V1)
        rows.append({"case": f"{site}:{lm}", "n_test_days": len(days_),
                     "targets": "; ".join(f"{a}: {P.setting_label(s)}" for a, s in RR["targets"].items()),
                     **dict(zip(A, Q)), "ranking": D.ranking_string(Q, A), "ranking_5crit": D.ranking_string(Q5, A)})
        if site == "sydney":
            OUT["sydney"] = {"n_test_days": len(days_), "S": S_.round(4).tolist(), "Q": Q.round(4).tolist(),
                             "kpi": kpi_rows(RR["sims"], RR["p"], len(days_) / 365.0).round(4).to_dict(orient="index"),
                             "targets": {a: P.setting_label(s) for a, s in RR["targets"].items()}}
            Qs_syd = D.bootstrap_Q(P.tensor(RR["blocks"])[0], OM, W0, B=1000, block=7, seed=SEED)
            OUT["sydney"]["bootstrap_p_first"] = [float((Qs_syd.argmax(1) == i).mean()) for i in range(NA)]
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "load_model_rankings.csv"), index=False)
    OUT["load_model_rankings"] = t.round(5).to_dict(orient="records")
    nm = {"zurich:stochastic": "Zurich, stochastic neighbourhood (baseline)", "zurich:h0": "Zurich, BDEW H0 standard profile",
          "zurich:metered": "Zurich, metered low-voltage aggregate", "sydney:measured": "Sydney, measured households and PV"}
    save_tex("load_model_rankings", [f"{nm[r['case']]} & " + " & ".join(f4(r[a]) for a in A) + " & " + rk(r["ranking"]) + " \\\\"
                                     for _, r in t.iterrows()])
    # benchmarks: cost-optimal perfect-foresight LPs and forecast-based LP controllers
    labs = A + BENCH
    Q, S, _, _ = P.base_Q(R["blocks"], alts=labs)
    OUT["Q_with_benchmarks"] = dict(zip(labs, np.round(Q, 4).tolist()))
    OUT["ranking_with_benchmarks"] = D.ranking_string(Q, labs)
    rnk = D.ranks(Q)
    lines = []
    for i, a in enumerate(labs):
        if a == "L2":
            lines.append("\\midrule")
        lines.append(f"${lab_tex(a)}$ {M.SHORT[a]} & " + " & ".join(f"{S[i, j]:.4f}" for j in range(NC)) + f" & {Q[i]:.4f} & {rnk[i]} \\\\")
    save_tex("benchmarks", lines)
    pd.DataFrame(np.c_[S, Q], index=labs, columns=CR + ["Q"]).to_csv(os.path.join(RES, "benchmarks.csv"))


# ============================================================== conventional crisp MCDM (R3.1)
def crisp_matrix(blocks, labels, p, day_idx=None):
    """Annual crisp decision matrix from block sums (optionally on resampled days)."""
    a_c = (1 - p["dr_rebound"]) + p["comfort_delta"] * p["dr_rebound"]
    out = []
    for a in labels:
        b = blocks[a]
        g = b.groupby("opday")
        day = pd.DataFrame({k: g[k].sum() for k in ["energy_cost", "served", "unmet", "adj", "re", "re_load", "ch_re",
                                                    "dis_re", "dr", "req"]})
        day["peak"] = g["peak"].max()
        if day_idx is not None:
            day = day.iloc[day_idx]
        t = day.sum()
        out.append([t.energy_cost / t.served, 1 - t.unmet / t.adj, (t.re_load + t.ch_re) / t.re,
                    (t.re_load + t.dis_re) / t.adj, day["peak"].mean(), 1 - a_c * t.dr / t.req])
    return np.array(out)


def section_conventional(R, X, S, stats):
    p = R["p"]; COST = np.array([j in (0, 4) for j in range(NC)])
    pool = A + BENCH
    Xc_pool = crisp_matrix(R["blocks"], pool, p)
    Xq_pool, _ = P.tensor(R["blocks"], pool)
    S_pool = D.scores_quantile(Xq_pool, OM)
    st_pool = coupling_cells(Xq_pool)
    Sfull_pool = np.array([[st_pool[(i, j)]["S"] for j in range(NC)] for i in range(len(pool))])
    SDe_pool = np.array([[st_pool[(i, j)]["sd"]["EMP"] for j in range(NC)] for i in range(len(pool))])
    L10e_pool = np.array([[st_pool[(i, j)]["L10"]["EMP"] for j in range(NC)] for i in range(len(pool))])
    # reference-point crisp annual matrix (same transforms as the DHFSS, no hesitancy, no periods)
    df = R["df"]; gmax = p["grid_max"]; imp = np.minimum(df.load_kw, gmax)
    rho = float((df.price * imp).sum() / imp.sum())
    Zref = Xc_pool.copy(); Zref[:, 0] = rho / (rho + Xc_pool[:, 0]); Zref[:, 4] = 1 - Xc_pool[:, 4] / gmax
    idx = {a: i for i, a in enumerate(pool)}
    methods = {
        "DHFSS: THFWA$^{\\perp}_{\\mathrm{AR}}$ + MABAC (baseline)": ("reference point", lambda L: D.mabac(S_pool[[idx[a] for a in L]], W0)[0]),
        "DHFSS: risk-averse $R^{\\mathrm{emp}}_{2}$": ("reference point", lambda L: D.wsm(Sfull_pool[[idx[a] for a in L]] - 2 * SDe_pool[[idx[a] for a in L]], W0)),
        "DHFSS: lower-tail $L^{\\mathrm{emp}}_{0.10}$ + MABAC": ("reference point", lambda L: D.mabac(L10e_pool[[idx[a] for a in L]], W0)[0]),
        "Crisp WSM": ("reference point", lambda L: D.wsm(Zref[[idx[a] for a in L]], W0)),
        "Crisp WSM (SAW)": ("min--max", lambda L: D.wsm_minmax(Xc_pool[[idx[a] for a in L]], W0, COST)),
        "Crisp MABAC (original)": ("min--max", lambda L: D.mabac_minmax(Xc_pool[[idx[a] for a in L]], W0, COST)),
        "TOPSIS": ("vector", lambda L: D.topsis_vector(Xc_pool[[idx[a] for a in L]], W0, COST)),
        "VIKOR ($v=0.5$)": ("linear (best--worst)", lambda L: D.vikor(Xc_pool[[idx[a] for a in L]], W0, COST)),
        "PROMETHEE II (V-shape)": ("pairwise, range", lambda L: D.promethee2(Xc_pool[[idx[a] for a in L]], W0, COST)),
    }
    base_scores = methods[list(methods)[0]][1](A)
    # bootstrap stability of the top rank (28-day moving blocks, same resamples as Table bootstrap)
    rng = np.random.default_rng(SEED); Dn = X.shape[-1]; B = 300
    boot_idx = []
    for _ in range(B):
        starts = rng.integers(0, Dn - 28 + 1, size=int(np.ceil(Dn / 28)))
        boot_idx.append(np.concatenate([np.arange(s, s + 28) for s in starts])[:Dn])
    days = sorted(set.intersection(*[set(R["blocks"][a].opday) for a in A]))
    blocks_d = {a: R["blocks"][a][R["blocks"][a].opday.isin(days)] for a in A}
    Xc_boot = [crisp_matrix(blocks_d, A, p, bi) for bi in boot_idx]
    rows = []
    for nm, (norm, fn) in methods.items():
        sc = fn(A)
        rr, n_tot = D.pairwise_reversal_rate(fn, pool)
        # bootstrap: recompute the method on resampled days
        tops = []
        for b, bi in enumerate(boot_idx):
            if nm.startswith("DHFSS"):
                Xb = X[..., bi]
                if "baseline" in nm:
                    s_b = D.mabac(D.scores_quantile(Xb, OM), W0)[0]
                else:
                    ev = np.einsum("k,ijkd->ijd", OM, Xb)                 # empirical (same-day) aggregates
                    if "risk" in nm:
                        s_b = D.wsm(ev.mean(-1) - 2 * ev.std(-1), W0)
                    else:
                        s_b = D.mabac(np.array([[D.lower_tail_mean(ev[i, j], 0.10) for j in range(NC)] for i in range(NA)]), W0)[0]
            else:
                Xc = Xc_boot[b]
                if nm == "Crisp WSM":
                    Zb = Xc.copy(); Zb[:, 0] = rho / (rho + Xc[:, 0]); Zb[:, 4] = 1 - Xc[:, 4] / gmax; s_b = D.wsm(Zb, W0)
                elif "SAW" in nm:
                    s_b = D.wsm_minmax(Xc, W0, COST)
                elif "MABAC" in nm:
                    s_b = D.mabac_minmax(Xc, W0, COST)
                elif nm == "TOPSIS":
                    s_b = D.topsis_vector(Xc, W0, COST)
                elif nm.startswith("VIKOR"):
                    s_b = D.vikor(Xc, W0, COST)
                else:
                    s_b = D.promethee2(Xc, W0, COST)
            tops.append(int(np.argmax(s_b)))
        top_full = int(np.argmax(sc))
        rows.append({"method": nm, "normalization": norm, "ranking": D.ranking_string(sc, A),
                     "kendall_tau": D.kendall_tau(sc, base_scores), "reversal_rate": rr, "n_subset_pairs": n_tot,
                     "boot_top_stability": float(np.mean(np.array(tops) == top_full)), "top": A[top_full]})
        print("  conventional:", nm, rows[-1]["ranking"], round(rr, 4), flush=True)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "conventional_mcdm.csv"), index=False)
    OUT["conventional_mcdm"] = t.round(4).to_dict(orient="records")
    OUT["conventional_pool"] = pool
    save_tex("conventional_mcdm", [f"{r['method']} & {r['normalization']} & {rk(r['ranking'])} & {r['kendall_tau']:.2f} & "
                                   f"{100*r['reversal_rate']:.1f} & {100*r['boot_top_stability']:.1f} \\\\" for _, r in t.iterrows()])


# ============================================================== joint sensitivity and preference regions (R3.5)
def section_joint(R, X, S):
    p = R["p"]; rb = p["dr_rebound"]
    rel = np.array([OUT["reliability_test"][a] for a in A])
    j6 = CR.index("C6")
    Xc6 = P.tensor(R["blocks"], A, ["C6_all"])[0][:, 0]                 # 1 - dr/req, (m, p, N)
    cshare = 1 - Xc6                                                     # curtailed share of requested energy
    Hq = np.nanquantile(X, (0.10, 0.50, 0.90), axis=-1).mean(axis=0)     # (m, n, p) quantile-HFE scores per period
    Hc = np.nanquantile(cshare, (0.10, 0.50, 0.90), axis=-1).mean(axis=0)  # (m, p)
    cov = {"IND": None, "EMP": None, "COM": None}
    m, n = NA, NC
    COV = np.zeros((3, m, n, 3, 3))
    for i in range(m):
        for j in range(n):
            x = cshare[i] if j == j6 else X[i, j]
            ce = np.cov(x, bias=True); cs = np.cov(np.sort(x, axis=1), bias=True)
            COV[0, i, j] = np.diag(np.diag(ce)); COV[1, i, j] = ce; COV[2, i, j] = cs
    rng = np.random.default_rng(SEED); N = 20_000
    Wc = rng.dirichlet(np.ones(n), N); Om = rng.dirichlet(np.ones(3), N)
    delta = rng.uniform(0, 1, N); beta = rng.uniform(0, 3, N); coup = rng.integers(0, 3, N)
    rreq = np.where(rng.uniform(size=N) < 0.5, 0.0, rng.uniform(0.985, 0.999, N))   # half: no requirement
    a_c = (1 - rb) + delta * rb                                          # C6 = 1 - a_c * cshare
    mu = np.einsum("nk,ijk->nij", Om, Hq)
    mu[:, :, j6] = 1 - a_c[:, None] * np.einsum("nk,ik->ni", Om, Hc)
    var = np.einsum("nk,cijkl,nl->ncij", Om, COV, Om)                    # (N, 3, m, n)
    var = var[np.arange(N), coup]                                        # (N, m, n)
    var[:, :, j6] *= a_c[:, None] ** 2
    sd = np.sqrt(np.clip(var, 0, None))
    score = np.einsum("nj,nij->ni", Wc, mu - beta[:, None, None] * sd)
    feas = rel[None, :] >= rreq[:, None]
    score = np.where(feas, score, -np.inf)
    win = np.where(feas.any(1), np.argmax(score, 1), -1)
    labels = np.array(A + ["none"])
    winner = labels[win]
    feat = pd.DataFrame(np.c_[Wc, Om, delta, beta, coup, rreq],
                        columns=[f"w_C{j+1}" for j in range(n)] + ["omega_offpeak", "omega_midpeak", "omega_onpeak",
                                                                  "delta", "beta", "coupling", "reliability_req"])
    feat["winner"] = winner
    feat.to_csv(os.path.join(RES, "joint_mc.csv"), index=False, float_format="%.5f")
    share = pd.Series(winner).value_counts(normalize=True).reindex(A + ["none"]).fillna(0)
    OUT["joint_winner_share"] = share.round(4).to_dict()
    # conditional winner shares by reliability requirement and comfort weight (interpretation aid)
    bins = {"reliability_req": [0.0, 0.5, 0.9875, 0.9900, 0.9950, 0.9975, 0.999], "w_C6": [0, 0.1, 0.2, 0.3, 1.0],
            "w_C1": [0, 0.1, 0.2, 0.3, 1.0], "beta": [0, 1, 2, 3]}
    cond = {}
    for f_, edges in bins.items():
        cut = pd.cut(feat[f_], edges, include_lowest=True)
        cond[f_] = (feat.groupby(cut, observed=True)["winner"].value_counts(normalize=True).unstack(fill_value=0)
                    .reindex(columns=A + ["none"], fill_value=0).round(3))
        cond[f_].to_csv(os.path.join(RES, f"joint_conditional_{f_}.csv"))
    OUT["joint_conditional"] = {k: {str(i): v.loc[i].to_dict() for i in v.index} for k, v in cond.items()}
    # decision tree: explicit conditions under which each policy is preferred
    from sklearn.tree import DecisionTreeClassifier, export_text
    Xf = feat.drop(columns=["winner"]).to_numpy(); yf = feat["winner"].to_numpy()
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=400, random_state=SEED).fit(Xf, yf)
    acc = float(tree.score(Xf, yf))
    rules = export_text(tree, feature_names=list(feat.columns[:-1]), decimals=3, show_weights=False)
    with open(os.path.join(RES, "joint_tree.txt"), "w", encoding="utf-8") as fh:
        fh.write(rules + f"\naccuracy {acc:.4f}\n")
    imp_ = dict(zip(feat.columns[:-1], np.round(tree.feature_importances_, 4)))
    OUT["joint_tree"] = {"accuracy": acc, "importance": imp_, "rules": rules}
    leaves = []
    tr = tree.tree_; fn = list(feat.columns[:-1])

    def walk(node, cond_):
        if tr.children_left[node] == -1:
            cnt = tr.value[node][0]; cnt = cnt / cnt.sum()
            k = int(np.argmax(cnt))
            leaves.append({"conditions": " and ".join(cond_) if cond_ else "all", "n": int(tr.n_node_samples[node]),
                           "share": float(tr.n_node_samples[node] / N), "winner": tree.classes_[k], "purity": float(cnt[k])})
            return
        f_, th_ = fn[tr.feature[node]], tr.threshold[node]
        walk(tr.children_left[node], cond_ + [f"{f_} <= {th_:.3f}"])
        walk(tr.children_right[node], cond_ + [f"{f_} > {th_:.3f}"])
    walk(0, [])
    lv = pd.DataFrame(leaves); lv.to_csv(os.path.join(RES, "joint_tree_leaves.csv"), index=False)
    OUT["joint_tree_leaves"] = lv.round(4).to_dict(orient="records")

    def texcond(c):
        rep = {"reliability_req": "$r_{\\min}$", "beta": "$\\beta$", "delta": "$\\delta$", "coupling": "coupling",
               "omega_offpeak": "$\\omega_1$", "omega_midpeak": "$\\omega_2$", "omega_onpeak": "$\\omega_3$"}
        for j in range(n):
            rep[f"w_C{j+1}"] = f"$w_{j+1}$"
        out = []
        le, gt = "$\\le$", "$>$"
        for part in c.split(" and "):
            f_, op, v = part.split(" ")
            sym = le if op == "<=" else gt
            if f_ == "coupling":
                out.append("coupling " + ("independent" if float(v) < 0.5 else "independent or empirical") if op == "<="
                           else "coupling " + ("empirical or comonotone" if float(v) < 0.5 else "comonotone"))
            else:
                out.append(f"{rep.get(f_, f_)} {sym} {float(v):.3f}")
        return "; ".join(out)
    save_tex("joint_tree", [f"{texcond(r['conditions'])} & {100*r['share']:.1f} & {rk(r['winner']) if r['winner'] != 'none' else 'none feasible'} & "
                            f"{100*r['purity']:.0f} \\\\" for _, r in lv.iterrows()])
    save_tex("joint_share", [" & ".join(f"{100*share[a]:.1f}" for a in A) + f" & {100*share['none']:.1f} \\\\"])
    return feat


def section_preference_map():
    from multiprocessing import Pool
    import gsa
    grid = [(g, r) for g in [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0] for r in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]]
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        rows = list(pool.imap(gsa.preference_eval, grid, chunksize=1))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "preference_map.csv"), index=False)
    OUT["preference_map_winner_share"] = t.winner.value_counts(normalize=True).round(4).to_dict()
    OUT["preference_map"] = t.round(5).to_dict(orient="records")
    return t


def oat_value_label(v):
    return "off" if v == "off" else (f"{v:g}" if isinstance(v, (int, float)) else str(v))


def section_oat():
    specs = [("Battery capacity (kWh)", "battery_kwh", [10.0, 20.0, 30.0, 40.0]),
             ("Grid-import limit (kW)", "grid_max", [10.0, 12.0, 14.0, 16.0, 18.0]),
             ("DR cap $\\rho^{\\max}$", "dr_cap", [0.05, 0.10, 0.15, 0.20]),
             ("DR rebound share", "dr_rebound", [0.0, 0.5, 1.0]),
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
            tg = RR["targets"]
            rows.append({"parameter": label, "value": v,
                         "targets": "/".join(("off" if s[1] == "off" else f"{100*float(s[1]):.0f}") + ("a" if s[0] == "any" else "")
                                             for s in (tg["A2"], tg["A4"], tg["A5"], tg["A6"])),
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


def section_extra_tables():
    """Supplementary tables built from summary.json and the joint-analysis files (Tables S11-S12)."""
    with open(os.path.join(RES, "summary.json"), encoding="utf-8") as fh:
        summ = json.load(fh)
    sy = summ["sydney"]; lines = []
    for i, a in enumerate(A):
        k = sy["kpi"][a]
        reo = f"{100*k['discharge_re_kwh']/k['discharge_kwh']:.0f}" if k["discharge_kwh"] > 0 else "--"
        lines.append(f"${lab_tex(a)}$ & {100*k['cost_per_served']:.2f} & {100*k['reliability']:.2f} & {100*k['service_ratio']:.2f} & {reo} & "
                     + " & ".join(f"{v:.4f}" for v in sy["S"][i]) + f" & {sy['Q'][i]:.4f} \\\\")
    save_tex("sydney_detail", lines)
    lines = []
    for f_, lab in (("reliability_req", "$r_{\\min}$"), ("w_C1", "$w_1$")):
        t = pd.read_csv(os.path.join(RES, f"joint_conditional_{f_}.csv"), index_col=0)
        for idx, r in t.iterrows():
            lo, hi = [float(x) for x in idx.strip("()[]").split(",")]
            if f_ == "reliability_req":
                cond = "no requirement" if hi <= 0.5 else f"{lab} in ({100*max(lo, 0.985):.2f}, {100*hi:.2f}]\\%"
            else:
                cond = f"{lab} in ({max(lo, 0):.2f}, {hi:.2f}]"
            lines.append(cond + " & " + " & ".join(f"{100*r[c]:.1f}" for c in A + ["none"]) + " \\\\")
        lines.append("\\midrule")
    save_tex("joint_conditional", lines[:-1])


def main():
    t0 = time.time()
    skip_gsa = "--skip-gsa" in sys.argv
    if "--figures-only" in sys.argv:
        import figures
        figures.from_results(); return
    def step(msg):
        print(f"[{time.time()-t0:6.0f} s] {msg}", flush=True)
    step("data"); section_data()
    step("baseline"); R = section_baseline()
    Q, S, X, days, H = section_scores(R)
    step("context/criteria"); section_context(H); section_criteria(R)
    step("hesitancy/operators/normalization"); section_hesitancy(X, S); section_operators(X, H); section_normalization(R, S)
    step("coupling"); cells, risk, stats = section_coupling(X, S)
    step("weights/temporal/sampling"); section_weights(S); section_temporal(X, R); section_sampling(X, days, S)
    step("load models, second site, benchmarks"); section_load_models_and_benchmarks(R)
    step("conventional MCDM"); section_conventional(R, X, S, stats)
    step("joint sensitivity"); section_joint(R, X, S)
    step("preference map"); section_preference_map()
    step("one-at-a-time"); section_oat()
    step("Morris"); section_gsa(skip_gsa)
    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(OUT, fh, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer, np.bool_)) else o.item())
    section_extra_tables()
    step("figures")
    import figures
    figures.from_results()
    print(json.dumps({k: OUT[k] for k in ("ranking", "Q", "targets", "bootstrap_p_first")}, default=str))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
