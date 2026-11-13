"""Figures (matplotlib). Every figure is drawn from computed results in results/."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")

SERIES = {"A1": "#2a78d6", "A2": "#eb6834", "A3": "#1baf7a", "A4": "#eda100", "A5": "#7b4fc9", "A6": "#d0457f",
          "L2": "#8c8b86", "L3": "#b9b8b3", "L4": "#52514e", "L2F": "#a9a39a", "L4F": "#6f6a62", "none": "#ffffff"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fc", "#9cc3ee", "#2a78d6", "#123f75"])
DIV = LinearSegmentedColormap.from_list("div", ["#b34a1f", "#eb6834", "#e9e8e4", "#2a78d6", "#123f75"])
# every symbol is typeset as LaTeX math (mathtext, Computer Modern)
SYM = {k: f"${k[0]}_{{{k[1:]}}}$" for k in ["A1", "A2", "A3", "A4", "A5", "A6", "L2", "L3", "L4", "L2F", "L4F",
                                             "C1", "C2", "C3", "C4", "C5", "C6"]}
SYM["none"] = "none"
POL = ["A1", "A2", "A3", "A4", "A5", "A6"]
NP = len(POL)
NAMES = {"A1": "$A_1$ Renewable-first", "A2": "$A_2$ Battery (cost-tuned)", "A3": "$A_3$ Price-elastic DR",
         "A4": "$A_4$ Hybrid (cost-tuned)", "A5": "$A_5$ Battery (MC-tuned)", "A6": "$A_6$ Hybrid (MC-tuned)",
         "L2": "$L_2$ LP battery", "L3": "$L_3$ LP DR", "L4": "$L_4$ LP battery + DR",
         "L2F": "$L_{2F}$ forecast LP battery", "L4F": "$L_{4F}$ forecast LP battery + DR"}
PER = ["t1_offpeak", "t2_midpeak", "t3_onpeak"]
PER_LAB = {"t1_offpeak": "$t_1$ off-peak", "t2_midpeak": "$t_2$ mid-peak", "t3_onpeak": "$t_3$ on-peak"}
CRIT = ["$C_1$ cost eff.", "$C_2$ reliability", "$C_3$ self-cons.", "$C_4$ RE-origin", "$C_5$ headroom", "$C_6$ comfort"]
NCR = len(CRIT)

# Latin Modern Roman (the LaTeX font) when available; DejaVu Serif otherwise
for _d in ("/usr/share/texmf/fonts/opentype/public/lm", "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm",
           "/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm"):
    if os.path.isdir(_d):
        for _f in os.listdir(_d):
            if _f.startswith("lmroman10-") and _f.endswith(".otf"):
                font_manager.fontManager.addfont(os.path.join(_d, _f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
                     "mathtext.fontset": "cm", "axes.unicode_minus": True, "axes.formatter.use_mathtext": True,
                     "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.8,
                     "pdf.fonttype": 42, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8.5})


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def _save(fig, name):
    """Save as 300-dpi PNG (journal submission format)."""
    fig.savefig(os.path.join(FIG, name), dpi=300, facecolor="white"); plt.close(fig)


def fig_inputs():
    import ems_data as E, pipeline as P
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.6, 1.2, 0.8]})
    ax = axes[0]
    cols = {"stochastic": SERIES["A1"], "h0": SERIES["A2"], "metered": SERIES["A3"]}
    labs = {"stochastic": "Stochastic (baseline)", "h0": "BDEW H0", "metered": "Metered LV"}
    daily = {}
    for lm in ["stochastic", "h0", "metered"]:
        d = P.inputs(lm); d = d[d.index.year.isin(E.TEST_YEARS)]
        h = d.groupby(d.index.hour).load_kw.mean(); ax.plot(h.index, h.values, color=cols[lm], label=labs[lm])
        daily[lm] = (d.load_kw.resample("D").sum() * 0.25).to_numpy()
    pv = d.groupby(d.index.hour).pv_kw.mean()
    ax.plot(pv.index, pv.values, color=SERIES["A4"], ls="--", label="PV (20 kWp)")
    try:
        ds = P.inputs(site="sydney")
        hs = ds.groupby(ds.index.hour).load_kw.mean()
        ax.plot(hs.index, hs.values, color=SERIES["A5"], ls=":", label="Sydney measured")
        daily["sydney"] = (ds.load_kw.resample("D").sum() * 0.25).to_numpy()
    except FileNotFoundError:
        pass
    ax.axvspan(17, 23, color="#fbeee6", lw=0); ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour (CET)"); ax.set_ylabel("Mean power (kW)"); _grid(ax); ax.set_ylim(top=15.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.6)
    ax.set_title("(a) Diurnal means (test periods)", loc="left", fontsize=9)
    ax = axes[1]
    keys = [k for k in ["stochastic", "h0", "metered", "sydney"] if k in daily]
    cols["sydney"] = SERIES["A5"]
    parts = ax.boxplot([daily[k] for k in keys], widths=0.55, patch_artist=True,
                       medianprops=dict(color=INK), whis=(2.5, 97.5), showfliers=False)
    for p_, k in zip(parts["boxes"], keys):
        p_.set_facecolor(cols[k]); p_.set_edgecolor("white")
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([{"stochastic": "Stoch.", "h0": "H0", "metered": "Metered", "sydney": "Sydney"}[k] for k in keys], fontsize=7.5)
    ax.set_ylabel("Daily energy (kWh)")
    _grid(ax); ax.set_title("(b) Daily demand", loc="left", fontsize=9)
    ax = axes[2]
    pr = d.groupby(d.index.hour).price.mean()
    ax.step(np.arange(25), np.r_[pr.values, pr.values[-1]], where="post", color=SERIES["A2"])
    ax.set_xlim(0, 24); ax.set_xticks([0, 7, 17, 23]); ax.set_ylim(0, 0.34); ax.set_ylabel("USD/kWh"); ax.set_xlabel("Hour")
    _grid(ax); ax.set_title("(c) Time-of-use tariff", loc="left", fontsize=9)
    fig.tight_layout(); _save(fig, "fig_inputs.png")


def fig_kpi():
    k = pd.read_csv(os.path.join(RES, "kpi_test.csv"), index_col=0)
    pol = list(k.index)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.0))
    share = (100 * k.discharge_re_kwh / k.discharge_kwh).where(k.discharge_kwh > 0)
    for ax, vals, lab in ((axes[0], k.energy_cost_usd / 1000, "Energy cost (kUSD/yr)"),
                          (axes[1], k.unmet_kwh, "Unserved energy (kWh/yr)"),
                          (axes[2], share, "Renewable-origin discharge (%)")):
        v = vals.to_numpy(float)
        ax.bar(range(len(pol)), np.nan_to_num(v), color=[SERIES[a] for a in pol], width=0.7, edgecolor="white")
        for x, y in enumerate(v):
            ax.text(x, (0 if np.isnan(y) else y), "n/a" if np.isnan(y) else (f"{y:.1f}" if y < 100 else f"{y:.0f}"),
                    ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(pol))); ax.set_xticklabels([SYM[a] for a in pol], fontsize=8.5); ax.set_title(lab, loc="left", fontsize=9)
        _grid(ax); ax.set_ylim(0, np.nanmax(v) * 1.2)
    fig.tight_layout(); _save(fig, "fig_kpi.png")


def fig_hfe(H):
    fig, axes = plt.subplots(3, NCR, figsize=(7.6, 4.8), sharex=True, sharey="col")
    for k in range(3):
        for j in range(NCR):
            ax = axes[k, j]
            for i, a in enumerate(POL):
                h = H[i][j][k]
                ax.plot([h[0], h[2]], [i, i], color=SERIES[a], lw=2.2, solid_capstyle="round")
                ax.plot(h[1], i, "o", color=SERIES[a], ms=3.6, mec="white", mew=0.6)
            ax.set_yticks(range(NP)); ax.set_yticklabels([SYM[a] for a in POL] if j == 0 else [], fontsize=8)
            ax.invert_yaxis(); _grid(ax, "x"); ax.tick_params(axis="x", labelsize=7)
            if k == 0:
                ax.set_title(CRIT[j], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(PER_LAB[PER[k]], fontsize=8)
    fig.text(0.5, 0.005, "Normalized grade (bar: $P_{10}$ to $P_{90}$; dot: $P_{50}$)", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); _save(fig, "fig_hfe.png")


def _heat(ax, M, cmap, vmin, vmax, fmt, rows, cols, dark_thr=0.62, fs=7.5):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color=INK2); continue
            rel = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            dark = rel > dark_thr or (cmap is DIV and (rel < 0.18 or rel > 0.82))
            ax.text(j, i, fmt.format(v).replace("-", "\u2212"), ha="center", va="center", fontsize=fs, color="white" if dark else INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_scores(S):
    import mcdm as D
    _, _, _, Qm = D.mabac(S, np.full(NCR, 1.0 / NCR))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    rows = [NAMES[a] for a in POL]
    im = _heat(axes[0], S, SEQ, 0, 1, "{:.3f}", rows, [SYM[f"C{j}"] for j in range(1, NCR + 1)])
    axes[0].set_title("(a) Aggregated scores $s_{ij}$", loc="left", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
    Qm = 1000 * Qm; v = np.abs(Qm).max()
    im = _heat(axes[1], Qm, DIV, -v, v, "{:+.1f}", [""] * NP, [SYM[f"C{j}"] for j in range(1, NCR + 1)])
    axes[1].set_title(r"(b) MABAC border distances $q_{ij}$ ($\times10^{-3}$)", loc="left", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    fig.tight_layout(); _save(fig, "fig_scores.png")


def _rank_row(Q):
    import mcdm as D
    Q = np.asarray(Q, float)
    r = np.full(len(Q), np.nan)
    ok = ~np.isnan(Q)
    r[ok] = D.ranks(Q[ok])
    return r


def _drivers_rows():
    """Every variant of the robustness protocol whose ranking is reported in the tables (R1.5)."""
    A = POL
    rows = []

    def add(group, label, Q):
        rows.append((group, label, _rank_row(Q)))
    cv = pd.read_csv(os.path.join(RES, "criterion_variants.csv"))
    short = {0: "Baseline ($C_1$\u2013$C_6$, equal weights)", 1: "First-submission $C_1$\u2013$C_5$", 2: "$C_2$ end-use service, no $C_6$",
             3: "$C_2$ end-use service and $C_6$", 4: "$C_6$ with $\\delta=0$", 5: "$C_6$ with $\\delta=1$", 6: "$C_1$ total cost",
             7: "$C_1$ incl. unserved-energy penalty", 8: "$C_1$ linear-clipped", 9: "$C_4$ counts grid-charged storage",
             10: "Penalty in $C_1$ and local $C_4$"}
    for k in range(len(cv)):
        lab = short.get(k, cv.iloc[k]["variant"].replace("leave out", "Leave out"))
        add("Criteria and comfort" if k < 11 else "Leave one criterion out", lab, cv.iloc[k][A].to_numpy(float))
    rc = pd.read_csv(os.path.join(RES, "reliability_constraint.csv"))
    for _, r in rc.iterrows():
        order = r["ranking"].split(" > ") if isinstance(r["ranking"], str) else []
        Q = np.full(NP, np.nan)
        for pos, a in enumerate(order):
            Q[A.index(a)] = len(order) - pos
        nm = "Reliability" if r["measure"] == "supply reliability" else "End-use service"
        add("Requirement (constraint)", f"{nm} $\\geq$ {100*r['threshold']:.2f}%", Q)
    cw = pd.read_csv(os.path.join(RES, "criterion_weights.csv"))
    for _, r in cw.iloc[1:].iterrows():
        add("Criterion weights", r["weights"].replace("ROC, ", "ROC "), r[A].to_numpy(float))
    tw = pd.read_csv(os.path.join(RES, "temporal_weights.csv"))
    for _, r in tw.iloc[1:].iterrows():
        add("Temporal weights", r["temporal"].replace(", $\\theta=0.5$", ""), r[A].to_numpy(float))
    hc = pd.read_csv(os.path.join(RES, "hfe_construction.csv"))
    hl = {"meansd:0": "Crisp means ($\\kappa=0$)", "bootstrap": "Bootstrap-mean quantiles", "quantile19": "19 quantiles"}
    for _, r in hc.iterrows():
        c = r["construction"]
        if c == "quantile":
            continue
        lab = hl[c] if c in hl else f"$\\mu\\pm{c.split(':')[1]}\\sigma$"
        add("Hesitancy (HFE construction)", lab, r[A].to_numpy(float))
    om = pd.read_csv(os.path.join(RES, "operators_methods.csv"))
    for _, r in om.iloc[1:].iterrows():
        cpl = {"IND": "\\perp", "COM": "\\top"}[r["operator"].split("-")[0]]; mean = r["operator"].split("-")[1]
        lab = f"$\\mathrm{{{mean}}}^{{{cpl}}}$" + (f", $\\varepsilon={r['eps']:g}$" if r["eps"] > 0 else "") + ("" if r["method"] == "MABAC" else f" + {r['method']}")
        add("Operator / ranking method", lab, r[A].to_numpy(float))
    nm_ = pd.read_csv(os.path.join(RES, "normalization.csv"))
    for s_, l in [("N1c", "Reference point, period totals"), ("N2", "Min–max, period totals"),
                  ("N3", "Linear ratio, period totals"), ("N4", "Interval-level min–max")]:
        add("Normalization", l, nm_[nm_.scheme == s_].iloc[0][A].to_numpy(float))
    rr = pd.read_csv(os.path.join(RES, "risk_rankings.csv"))
    cl = {"IND": "$\\perp$", "EMP": "emp", "COM": "$\\top$"}
    sl = {"S-0.5sd": "$S-0.5\\sigma$", "S-1sd": "$S-\\sigma$", "S-2sd": "$S-2\\sigma$", "S-4sd": "$S-4\\sigma$",
          "L10": "$L_{0.10}$", "L25": "$L_{0.25}$"}
    for _, r in rr.iterrows():
        if r["score"] == "S-0sd":
            continue
        add("Risk-averse score / coupling", f"{sl[r['score']]}, {cl[r['coupling']]}", r[A].to_numpy(float))
    lm = pd.read_csv(os.path.join(RES, "load_model_rankings.csv"))
    for c, l in [("zurich:h0", "Zurich, BDEW H0 load"), ("zurich:metered", "Zurich, metered LV load"),
                 ("sydney:measured", "Sydney, measured load and PV")]:
        add("Load model / site", l, lm[lm.case == c].iloc[0][A].to_numpy(float))
    py = pd.read_csv(os.path.join(RES, "per_year.csv"))
    for _, r in py.iterrows():
        add("Test year", f"{int(r['year'])}", r[A].to_numpy(float))
    return rows


def _drivers_panel(ax, rows):
    disp_labels, disp_vals, is_head = [], [], []
    prev = None
    for g, l, r in rows:
        if g != prev:
            disp_labels.append(g); disp_vals.append(np.full(NP, np.nan)); is_head.append(True); prev = g
        disp_labels.append(l); disp_vals.append(r); is_head.append(False)
    M = np.array(disp_vals)
    cmap = ListedColormap(["#123f75", "#2a78d6", "#6fa6e6", "#9cc3ee", "#cfe1f6", "#f3f7fc"]).with_extremes(bad="white")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.5, vmax=NP + 0.5, aspect="auto")
    for i in range(M.shape[0]):
        if is_head[i]:
            continue
        for j in range(NP):
            v = M[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{int(v)}", ha="center", va="center", fontsize=6.3,
                    color=("white" if (not np.isnan(v) and v <= 3) else INK))
    ax.set_xticks(range(NP)); ax.set_xticklabels([SYM[a] for a in POL], fontsize=8); ax.xaxis.tick_top()
    ax.set_yticks(range(len(disp_labels))); ax.set_yticklabels(disp_labels, fontsize=6.3)
    for tl, hd in zip(ax.get_yticklabels(), is_head):
        if hd:
            tl.set_fontweight("bold"); tl.set_color(INK)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)


def fig_drivers():
    rows = _drivers_rows()
    groups = list(dict.fromkeys(g for g, _, _ in rows))
    # split into two columns of similar height without breaking a group
    sizes = {g: 1 + sum(1 for x in rows if x[0] == g) for g in groups}
    tot = sum(sizes.values()); acc = 0; left = []
    for g in groups:
        if acc + sizes[g] / 2 <= tot / 2:
            left.append(g); acc += sizes[g]
    r1 = [x for x in rows if x[0] in left]; r2 = [x for x in rows if x[0] not in left]
    n = max(len(r1) + len(left), len(r2) + len(groups) - len(left))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 0.112 * n + 0.7))
    _drivers_panel(axes[0], r1); _drivers_panel(axes[1], r2)
    for ax, rr in zip(axes, (r1, r2)):
        k = len(rr) + len(dict.fromkeys(g for g, _, _ in rr))
        ax.set_ylim(n - 0.5, -0.5)
    fig.tight_layout(w_pad=0.6); _save(fig, "fig_drivers.png")


def fig_coupling(cells, risk):
    import json
    with open(os.path.join(RES, "summary.json"), encoding="utf-8") as fh:
        pair = json.load(fh).get("coupling_pair", "A1-A3")
    a_hi, a_lo = pair.split("-")
    c = cells.dropna(subset=["pos"]).copy()
    c["lab"] = [f"{SYM[a]}, {SYM[k]}" for a, k in zip(c["alt"], c["crit"])]
    c = c.sort_values(["crit", "alt"])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]; y = np.arange(len(c))
    ax.hlines(y, c.sd_IND, c.sd_upper, color=GRID, lw=3)
    ax.plot(c.sd_IND, y, "o", color=SERIES["A1"], ms=3.6, label="independent ($\\perp$)")
    ax.plot(c.sd_EMP, y, "D", color=SERIES["A3"], ms=3.2, label="empirical (same day)")
    ax.plot(c.sd_COM, y, "s", color=SERIES["A2"], ms=3.2, label="comonotone ($\\top$)")
    ax.plot(c.sd_upper, y, "|", color=INK2, ms=6, label=r"bound $\sum\omega_k\sigma_k$")
    ax.set_yticks(y); ax.set_yticklabels(c.lab, fontsize=6.5); ax.invert_yaxis(); _grid(ax, "x")
    ax.set_xlabel(r"Dispersion $\sigma$ of the aggregated HFE"); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("(a) Coupling and dispersion", loc="left", fontsize=9)
    ax = axes[1]
    thr = pd.read_csv(os.path.join(RES, "reversal_thresholds.csv"))
    sub = thr[thr.pair == pair].set_index("coupling")
    bgrid = np.linspace(0, 5, 251)
    for cp, ls, col in [("IND", "-", SERIES["A1"]), ("EMP", "--", SERIES["A3"]), ("COM", ":", SERIES["A2"])]:
        dW, dD = sub.loc[cp, "dW"], sub.loc[cp, "dD"]
        gap = -(dW - bgrid * dD)
        lab = {"IND": "independent", "EMP": "empirical", "COM": "comonotone"}[cp]
        b = sub.loc[cp, "beta_star"]
        bl = r"$\beta^\ast=\infty$" if not np.isfinite(b) else (r"$\beta^\ast=" + f"{b:.2f}" + "$")
        ax.plot(bgrid, gap * 1000, ls=ls, color=col, lw=1.8, label=f"{lab} ({bl})")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(r"Risk aversion $\beta$")
    ax.set_ylabel(f"$R_\\beta({a_lo[0]}_{a_lo[1]})-R_\\beta({a_hi[0]}_{a_hi[1]})$ ($\\times10^{{-3}}$)")
    _grid(ax); ax.legend(fontsize=7, loc="upper left")
    yl = ax.get_ylim()
    ax.text(0.98, 0.97, f"{SYM[a_lo]} ahead", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    ax.text(0.98, 0.50, f"{SYM[a_hi]} ahead", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    ax.set_title(f"(b) Coupling and the {SYM[a_hi]} vs. {SYM[a_lo]} decision", loc="left", fontsize=9)
    fig.tight_layout(); _save(fig, "fig_coupling.png")


def fig_robust(b_flat, Qs, per_year):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    rows = POL
    _heat(axes[0], b_flat, SEQ, 0, 1, "{:.2f}", [SYM[a] for a in rows], [f"$r={r}$" for r in range(1, NP + 1)], fs=6.5)
    axes[0].set_title("(a) SMAA $b_i^r$ (flat weights)", loc="left", fontsize=9)
    ax = axes[1]
    _orient = {"orientation": "horizontal"} if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 10) else {"vert": False}
    parts = ax.boxplot([Qs[:, i] for i in range(NP)], **_orient, widths=0.5, patch_artist=True,
                       medianprops=dict(color=INK, lw=1.1), whis=(2.5, 97.5), showfliers=False)
    for p_, a in zip(parts["boxes"], rows):
        p_.set_facecolor(SERIES[a]); p_.set_edgecolor("white")
    ax.set_yticks(range(1, NP + 1)); ax.set_yticklabels([SYM[a] for a in rows]); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.7, ls=":")
    _grid(ax, "x"); ax.set_xlabel("$Q$ (28-day block bootstrap)"); ax.set_title("(b) Sampling", loc="left", fontsize=9)
    ax = axes[2]
    for a in rows:
        ax.plot(per_year["year"], per_year[a], "o-", color=SERIES[a], ms=3.2, label=SYM[a])
    ax.set_xticks(per_year["year"]); _grid(ax); ax.set_ylabel("$Q$"); ax.legend(fontsize=6.5, ncol=3, loc="center left", bbox_to_anchor=(0, 0.42))
    ax.set_title("(c) Test years", loc="left", fontsize=9)
    fig.tight_layout(); _save(fig, "fig_robust.png")


def fig_gsa(g, idx):
    names = {"grid_max": "grid limit", "c_unmet": "penalty", "battery_kwh": "battery", "dr_cap": "DR cap $\\rho^{\\max}$",
             "tariff_ratio": "tariff ratio"}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, o, title in ((axes[0], "gap_A5_A6", "(a) $Q(A_5)-Q(A_6)$"),
                         (axes[1], "lead_storage", "(b) Lead of tuned storage")):
        sub = idx[idx.output == o]
        ax.scatter(sub.mu_star, sub.sigma, color=SERIES["A1"], s=18, zorder=3)
        tiny = sub[(sub.mu_star < 1e-6) & (sub.sigma < 1e-6)]
        for _, r in sub.drop(tiny.index).iterrows():
            ax.annotate(names[r.factor], (r.mu_star, r.sigma), fontsize=7, xytext=(3, 2), textcoords="offset points")
        if len(tiny):
            ax.annotate(", ".join(names[f] for f in tiny.factor) + " (no effect)", (0, 0), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        _grid(ax, "both"); ax.set_xlabel(r"$\mu^\ast$"); ax.set_ylabel(r"$\sigma$"); ax.set_title(title, loc="left", fontsize=9)
        lim = max(sub.mu_star.max(), sub.sigma.max()) * 1.3
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax = axes[2]
    share = g["winner"].value_counts(normalize=True).reindex(POL).fillna(0)
    ax.bar(range(NP), share.values, color=[SERIES[a] for a in share.index], width=0.7, edgecolor="white")
    for x, v in enumerate(share.values):
        ax.text(x, v, f"{100*v:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(NP)); ax.set_xticklabels([SYM[a] for a in share.index], fontsize=8); ax.set_ylim(0, 1.12); _grid(ax)
    ax.set_title("(c) Winner share", loc="left", fontsize=9)
    fig.tight_layout(); _save(fig, "fig_gsa.png")


def fig_oat(oat):
    params = list(dict.fromkeys(oat["parameter"]))
    n = len(params); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.6, 2.0 * nrow), sharey=True)
    axes = axes.ravel()
    for ax, par in zip(axes, params):
        g = oat[oat["parameter"] == par]; x = np.arange(len(g))
        for a in POL:
            ax.plot(x, g[a].to_numpy(float), color=SERIES[a], marker="o", ms=3.2, label=a)
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in g["value"]], fontsize=7.5)
        ax.set_title(par, loc="left", fontsize=8); _grid(ax); ax.axhline(0, color=INK2, lw=0.6, ls=":")
    for ax in axes[n:]:
        ax.axis("off")
    for r in range(nrow):
        axes[r * ncol].set_ylabel("$Q$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [NAMES[x] for x in l], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); _save(fig, "fig_oat.png")


def fig_joint():
    """Preference regions (R3.5): (a) physical map with re-tuning, (b)-(c) decision-parameter Monte Carlo."""
    pm = pd.read_csv(os.path.join(RES, "preference_map.csv"))
    jm = pd.read_csv(os.path.join(RES, "joint_mc.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    gl = sorted(pm.grid_max.unique()); tr = sorted(pm.tariff_ratio.unique())
    for _, r in pm.iterrows():
        x = tr.index(r.tariff_ratio); y = gl.index(r.grid_max)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=SERIES[r.winner], ec="white", lw=1))
        ax.text(x, y, f"{1000*r.margin:.1f}", ha="center", va="center", fontsize=6.3, color="white")
    ax.set_xlim(-0.5, len(tr) - 0.5); ax.set_ylim(-0.5, len(gl) - 0.5)
    ax.set_xticks(range(len(tr))); ax.set_xticklabels([f"{v:g}" for v in tr], fontsize=7.5)
    ax.set_yticks(range(len(gl))); ax.set_yticklabels([f"{v:g}" for v in gl], fontsize=7.5)
    ax.set_xlabel("On-/off-peak price ratio"); ax.set_ylabel("Grid-import limit (kW)")
    wins = [a for a in POL if a in set(pm.winner)]
    for a in wins:
        ax.plot([], [], "s", color=SERIES[a], label=SYM[a])
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(wins), handletextpad=0.2, columnspacing=0.8)
    ax.set_title("(a) Winner and margin ($\\times10^{-3}$)", loc="left", fontsize=9)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    labels = POL + ["none"]
    for ax, col, edges, ticks, title, xl in (
            (axes[1], "reliability_req", [-0.001, 0.5, 0.9875, 0.990, 0.9925, 0.995, 0.9975, 0.999],
             ["none", "98.5–\n98.75", "–99.0", "–99.25", "–99.5", "–99.75", "–99.9"],
             "(b) Reliability requirement", "$r_{\\min}$ (%)"),
            (axes[2], "w_C1", [0, 0.1, 0.2, 0.3, 0.45, 1.0], ["0–0.1", "–0.2", "–0.3", "–0.45", ">0.45"],
             "(c) Weight on cost $w_1$", "$w_1$")):
        cut = pd.cut(jm[col], edges, include_lowest=True)
        sh = jm.groupby(cut, observed=True)["winner"].value_counts(normalize=True).unstack(fill_value=0).reindex(columns=labels, fill_value=0)
        bottom = np.zeros(len(sh))
        for a in labels:
            v = sh[a].to_numpy()
            if v.sum() == 0:
                continue
            ax.bar(range(len(sh)), v, bottom=bottom, color=SERIES[a], width=0.8, edgecolor=("#bbbbbb" if a == "none" else "white"),
                   lw=0.5, label=SYM[a])
            bottom += v
        ax.set_xticks(range(len(sh))); ax.set_xticklabels(ticks[:len(sh)], fontsize=6.3); ax.set_ylim(0, 1)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("Winner share", fontsize=8); ax.set_title(title, loc="left", fontsize=9)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.66, 1.06), fontsize=7, handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); _save(fig, "fig_joint.png")


def make_all(R, S, X, H, cells, risk, b_flat, Qs, per_year, oat, gsa_res, gsa_idx):
    fig_inputs(); fig_kpi(); fig_hfe(H); fig_scores(S); fig_drivers(); fig_coupling(cells, risk)
    fig_robust(b_flat, Qs, per_year); fig_gsa(gsa_res, gsa_idx); fig_oat(oat); fig_joint()


def from_results():
    """Regenerate every figure from the files written to results/ by run_all.py."""
    import json
    import mcdm as D
    S = pd.read_csv(os.path.join(RES, "scores.csv"), index_col=0).to_numpy()
    X = np.load(os.path.join(RES, "block_tensor_test.npy"))
    H = D.build_hfes(X, "quantile")
    cells = pd.read_csv(os.path.join(RES, "coupling_cells.csv"))
    risk = pd.read_csv(os.path.join(RES, "risk_rankings.csv"))
    with open(os.path.join(RES, "summary.json"), encoding="utf-8") as fh:
        summ = json.load(fh)
    b_flat = np.array(summ["smaa_flat"])
    Qs = np.load(os.path.join(RES, "bootstrap_Q.npy"))
    per_year = pd.read_csv(os.path.join(RES, "per_year.csv"))
    oat = pd.read_csv(os.path.join(RES, "oat.csv"))
    g = pd.read_csv(os.path.join(RES, "gsa_morris.csv")); idx = pd.read_csv(os.path.join(RES, "gsa_indices.csv"))
    make_all(None, S, X, H, cells, risk, b_flat, Qs, per_year, oat, g, idx)
