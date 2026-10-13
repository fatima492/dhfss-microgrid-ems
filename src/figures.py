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

SERIES = {"A1": "#2a78d6", "A2": "#eb6834", "A3": "#1baf7a", "A4": "#eda100",
          "L2": "#8c8b86", "L3": "#b9b8b3", "L4": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fc", "#9cc3ee", "#2a78d6", "#123f75"])
DIV = LinearSegmentedColormap.from_list("div", ["#b34a1f", "#eb6834", "#e9e8e4", "#2a78d6", "#123f75"])
# every symbol is typeset as LaTeX math (mathtext, Computer Modern)
SYM = {k: f"${k[0]}_{k[1]}$" for k in ["A1", "A2", "A3", "A4", "L2", "L3", "L4", "C1", "C2", "C3", "C4", "C5"]}
POL = ["A1", "A2", "A3", "A4"]
NAMES = {"A1": "$A_1$ Renewable-first", "A2": "$A_2$ Battery", "A3": "$A_3$ Price-elastic DR", "A4": "$A_4$ Hybrid",
         "L2": "$L_2$ LP battery", "L3": "$L_3$ LP DR", "L4": "$L_4$ LP battery + DR"}
PER = ["t1_offpeak", "t2_midpeak", "t3_onpeak"]
PER_LAB = {"t1_offpeak": "$t_1$ off-peak", "t2_midpeak": "$t_2$ mid-peak", "t3_onpeak": "$t_3$ on-peak"}
CRIT = ["$C_1$ cost eff.", "$C_2$ reliability", "$C_3$ self-cons.", "$C_4$ RE-origin", "$C_5$ headroom"]

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
    ax.axvspan(17, 23, color="#fbeee6", lw=0); ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour (CET)"); ax.set_ylabel("Mean power (kW)"); _grid(ax); ax.set_ylim(top=15.5)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.6)
    ax.set_title("(a) Diurnal means, 2018\u20132021", loc="left", fontsize=9)
    ax = axes[1]
    parts = ax.boxplot([daily[k] for k in ["stochastic", "h0", "metered"]], widths=0.55, patch_artist=True,
                       medianprops=dict(color=INK), whis=(2.5, 97.5), showfliers=False)
    for p_, k in zip(parts["boxes"], ["stochastic", "h0", "metered"]):
        p_.set_facecolor(cols[k]); p_.set_edgecolor("white")
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Stochastic", "H0", "Metered"]); ax.set_ylabel("Daily energy (kWh)")
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
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
    share = (100 * k.discharge_re_kwh / k.discharge_kwh).where(k.discharge_kwh > 0)
    for ax, vals, lab in ((axes[0], k.energy_cost_usd / 1000, "Energy cost (kUSD/yr)"),
                          (axes[1], k.unmet_kwh, "Unserved energy (kWh/yr)"),
                          (axes[2], share, "Renewable-origin discharge (%)")):
        v = vals.to_numpy(float)
        ax.bar(range(len(pol)), np.nan_to_num(v), color=[SERIES[a] for a in pol], width=0.7, edgecolor="white")
        for x, y in enumerate(v):
            ax.text(x, (0 if np.isnan(y) else y), "n/a" if np.isnan(y) else (f"{y:.1f}" if y < 100 else f"{y:.0f}"),
                    ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(pol))); ax.set_xticklabels([SYM[a] for a in pol], fontsize=8.5); ax.set_title(lab, loc="left", fontsize=9)
        _grid(ax); ax.set_ylim(0, np.nanmax(v) * 1.2)
    fig.tight_layout(); _save(fig, "fig_kpi.png")


def fig_hfe(H):
    fig, axes = plt.subplots(3, 5, figsize=(7.4, 4.2), sharex=True, sharey="col")
    for k in range(3):
        for j in range(5):
            ax = axes[k, j]
            for i, a in enumerate(["A1", "A2", "A3", "A4"]):
                h = H[i][j][k]
                ax.plot([h[0], h[2]], [i, i], color=SERIES[a], lw=2.2, solid_capstyle="round")
                ax.plot(h[1], i, "o", color=SERIES[a], ms=4, mec="white", mew=0.6)
            ax.set_yticks(range(4)); ax.set_yticklabels([SYM[a] for a in POL] if j == 0 else [], fontsize=8.5)
            ax.invert_yaxis(); _grid(ax, "x"); ax.tick_params(axis="x", labelsize=7.5)
            if k == 0:
                ax.set_title(CRIT[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(PER_LAB[PER[k]], fontsize=8)
    fig.text(0.5, 0.005, "Normalized grade (bar: $P_{10}$ to $P_{90}$; dot: $P_{50}$)", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); _save(fig, "fig_hfe.png")


def _heat(ax, M, cmap, vmin, vmax, fmt, rows, cols, dark_thr=0.62):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color=INK2); continue
            rel = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            dark = rel > dark_thr or (cmap is DIV and (rel < 0.18 or rel > 0.82))
            ax.text(j, i, fmt.format(v).replace("-", "\u2212"), ha="center", va="center", fontsize=8, color="white" if dark else INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_scores(S):
    import mcdm as D
    _, _, _, Qm = D.mabac(S, np.full(5, 0.2))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.3))
    rows = [NAMES[a] for a in ["A1", "A2", "A3", "A4"]]
    im = _heat(axes[0], S, SEQ, 0, 1, "{:.3f}", rows, [SYM[f"C{j}"] for j in range(1, 6)])
    axes[0].set_title("(a) Aggregated scores $s_{ij}$", loc="left", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
    Qm = 1000 * Qm; v = np.abs(Qm).max()
    im = _heat(axes[1], Qm, DIV, -v, v, "{:+.1f}", [""] * 4, [SYM[f"C{j}"] for j in range(1, 6)])
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


def fig_drivers():
    A = ["A1", "A2", "A3", "A4"]
    rows = []
    def add(group, label, Q):
        rows.append((group, label, _rank_row(Q)))
    cv = pd.read_csv(os.path.join(RES, "criterion_variants.csv"))
    add("Baseline", "Equal weights, duration temporal weights", cv.iloc[0][A].to_numpy(float))
    lab = {1: "$C_1$ total cost (no penalty)", 2: "$C_1$ incl. unserved-energy penalty", 3: "$C_1$ linear-clipped $1-y/\\rho$",
           4: "$C_4$ counts grid-charged storage", 5: "Penalty in $C_1$ and local $C_4$"}
    for k, l in lab.items():
        add("Criterion design", l, cv.iloc[k][A].to_numpy(float))
    for j in range(5):
        add("Criterion design", f"Leave out $C_{j+1}$", cv.iloc[6 + j][A].to_numpy(float))
    rc = pd.read_csv(os.path.join(RES, "reliability_constraint.csv"))
    for _, r in rc.iterrows():
        order = r["ranking"].split(" > ") if isinstance(r["ranking"], str) else []; Q = np.full(4, np.nan)
        for pos, a in enumerate(order):
            Q[A.index(a)] = len(order) - pos
        add("Reliability constraint", f"Reliability $\\geq$ {100*r['threshold']:.2f}%", Q)
    cw = pd.read_csv(os.path.join(RES, "criterion_weights.csv"))
    for _, r in cw.iloc[1:].iterrows():
        nm = r["weights"]
        if nm.startswith("Stated vector"):
            nm = "Stated priority vector"
        add("Criterion weights", nm, r[A].to_numpy(float))
    tw = pd.read_csv(os.path.join(RES, "temporal_weights.csv"))
    for _, r in tw.iloc[1:].iterrows():
        nm = r["temporal"]
        add("Temporal weights", nm, r[A].to_numpy(float))
    hc = pd.read_csv(os.path.join(RES, "hfe_construction.csv"))
    for c, l in [("meansd:0", "Crisp means ($\\kappa=0$)"), ("meansd:2.0", "$\\{\\mu-2\\sigma,\\mu,\\mu+2\\sigma\\}$"),
                 ("quantile19", "$\\{P_{5},P_{10},\\dots,P_{95}\\}$")]:
        add("Hesitancy", l, hc[hc.construction == c].iloc[0][A].to_numpy(float))
    om = pd.read_csv(os.path.join(RES, "operators_methods.csv"))
    for (op, eps, me), l in [(("IND-EIN", 0.0, "MABAC"), "$\\mathrm{THFWA}^{\\perp}_{\\mathrm{EIN}}$, $\\varepsilon=0$"),
                             (("IND-ALG", 0.0, "MABAC"), "$\\mathrm{THFWA}^{\\perp}_{\\mathrm{ALG}}$, $\\varepsilon=0$"),
                             (("IND-ALG", 0.01, "MABAC"), "$\\mathrm{THFWA}^{\\perp}_{\\mathrm{ALG}}$, $\\varepsilon=0.01$"),
                             (("COM-ALG", 0.01, "MABAC"), "$\\mathrm{THFWA}^{\\top}_{\\mathrm{ALG}}$, $\\varepsilon=0.01$"),
                             (("IND-AR", 0.0, "TOPSIS"), "TOPSIS on scores $s_{ij}$"), (("IND-AR", 0.0, "HF-TOPSIS"), "Hesitant fuzzy TOPSIS")]:
        sub = om[(om.operator == op) & (np.isclose(om.eps, eps)) & (om.method == me)]
        add("Operator / method", l, sub.iloc[0][A].to_numpy(float))
    nm_ = pd.read_csv(os.path.join(RES, "normalization.csv"))
    for s, l in [("N1c", "Reference point, period totals"), ("N2", "Min\u2013max, period totals"),
                 ("N3", "Linear ratio, period totals"), ("N4", "Interval-level min\u2013max")]:
        add("Normalization", l, nm_[nm_.scheme == s].iloc[0][A].to_numpy(float))
    rr = pd.read_csv(os.path.join(RES, "risk_rankings.csv"))
    for sc, cp, l in [("S-2sd", "IND", "$S-2\\sigma$, independent ($\\perp$)"), ("S-2sd", "COM", "$S-2\\sigma$, comonotone ($\\top$)"),
                      ("L10", "EMP", "$L_{0.10}$, empirical"), ("L10", "COM", "$L_{0.10}$, comonotone ($\\top$)")]:
        sub = rr[(rr.score == sc) & (rr.coupling == cp)]
        add("Risk-averse score", l, sub.iloc[0][A].to_numpy(float))
    lm = pd.read_csv(os.path.join(RES, "load_model_rankings.csv"))
    for c, l in [("load:h0", "BDEW H0 load"), ("load:metered", "Metered LV load")]:
        add("Load model", l, lm[lm.case == c].iloc[0][A].to_numpy(float))
    disp_labels, disp_vals, is_head = [], [], []
    prev = None
    for g, l, r in rows:
        if g != prev:
            disp_labels.append(g); disp_vals.append(np.full(4, np.nan)); is_head.append(True); prev = g
        disp_labels.append(l); disp_vals.append(r); is_head.append(False)
    M = np.array(disp_vals)
    h = 0.155 * len(disp_labels) + 0.6
    fig, ax = plt.subplots(figsize=(4.6, h))
    cmap = ListedColormap(["#123f75", "#2a78d6", "#9cc3ee", "#f3f7fc"]).with_extremes(bad="white")
    Mplot = np.ma.masked_invalid(M)
    ax.imshow(Mplot, cmap=cmap, vmin=0.5, vmax=4.5, aspect="auto")
    for i in range(M.shape[0]):
        if is_head[i]:
            continue
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, "--" if np.isnan(v) else f"{int(v)}", ha="center", va="center", fontsize=7.5,
                    color=("white" if (not np.isnan(v) and v <= 2) else INK))
    ax.set_xticks(range(4)); ax.set_xticklabels([SYM[a] for a in POL], fontsize=9); ax.xaxis.tick_top()
    ax.set_yticks(range(len(disp_labels)))
    ax.set_yticklabels(disp_labels, fontsize=7.5)
    for tl, hd in zip(ax.get_yticklabels(), is_head):
        if hd:
            tl.set_fontweight("bold"); tl.set_color(INK)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(); _save(fig, "fig_drivers.png")


def fig_coupling(cells, risk):
    c = cells.dropna(subset=["pos"]).copy()
    c["lab"] = [f"{SYM[a]}, {SYM[k]}" for a, k in zip(c["alt"], c["crit"])]
    c = c.sort_values(["crit", "alt"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]; y = np.arange(len(c))
    ax.hlines(y, c.sd_IND, c.sd_upper, color=GRID, lw=3)
    ax.plot(c.sd_IND, y, "o", color=SERIES["A1"], ms=4, label="independent ($\\perp$)")
    ax.plot(c.sd_EMP, y, "D", color=SERIES["A3"], ms=3.5, label="empirical (same day)")
    ax.plot(c.sd_COM, y, "s", color=SERIES["A2"], ms=3.5, label="comonotone ($\\top$)")
    ax.plot(c.sd_upper, y, "|", color=INK2, ms=7, label=r"bound $\sum\omega_k\sigma_k$")
    ax.set_yticks(y); ax.set_yticklabels(c.lab, fontsize=7.5); ax.invert_yaxis(); _grid(ax, "x")
    ax.set_xlabel(r"Dispersion $\sigma$ of the aggregated HFE"); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_title("(a) Coupling and dispersion", loc="left", fontsize=9)
    ax = axes[1]
    thr = pd.read_csv(os.path.join(RES, "reversal_thresholds.csv"))
    sub = thr[thr.pair == "A1-A3"].set_index("coupling")
    bgrid = np.linspace(0, 4, 201)
    for cp, ls, col in [("IND", "-", SERIES["A1"]), ("EMP", "--", SERIES["A3"]), ("COM", ":", SERIES["A2"])]:
        dW, dD = sub.loc[cp, "dW"], sub.loc[cp, "dD"]
        gap = -(dW - bgrid * dD)                       # R(A3) - R(A1)
        lab = {"IND": "independent", "EMP": "empirical", "COM": "comonotone"}[cp]
        ax.plot(bgrid, gap * 1000, ls=ls, color=col, lw=1.8, label=f"{lab} ($\\beta^\\ast={sub.loc[cp, 'beta_star']:.2f}$)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(r"Risk aversion $\beta$"); ax.set_ylabel(r"$R_\beta(A_3)-R_\beta(A_1)$ ($\times10^{-3}$)")
    _grid(ax); ax.legend(fontsize=7.5, loc="upper left")
    ax.text(0.05, 0.12, "$A_3$ ahead", ha="left", va="bottom", fontsize=7.5, color=INK2)
    ax.text(0.05, -0.12, "$A_1$ ahead", ha="left", va="top", fontsize=7.5, color=INK2)
    ax.set_title("(b) Coupling and the $A_1$ vs. $A_3$ decision", loc="left", fontsize=9)
    fig.tight_layout(); _save(fig, "fig_coupling.png")


def fig_robust(b_flat, Qs, per_year):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    rows = POL
    _heat(axes[0], b_flat, SEQ, 0, 1, "{:.2f}", [SYM[a] for a in rows], [f"$r={r}$" for r in range(1, 5)])
    axes[0].set_title("(a) SMAA $b_i^r$ (flat)", loc="left", fontsize=9)
    ax = axes[1]
    _orient = {"orientation": "horizontal"} if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 10) else {"vert": False}
    parts = ax.boxplot([Qs[:, i] for i in range(4)], **_orient, widths=0.5, patch_artist=True,
                       medianprops=dict(color=INK, lw=1.1), whis=(2.5, 97.5), showfliers=False)
    for p_, a in zip(parts["boxes"], rows):
        p_.set_facecolor(SERIES[a]); p_.set_edgecolor("white")
    ax.set_yticks(range(1, 5)); ax.set_yticklabels([SYM[a] for a in rows]); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.7, ls=":")
    _grid(ax, "x"); ax.set_xlabel("$Q$ (28-day block bootstrap)"); ax.set_title("(b) Sampling", loc="left", fontsize=9)
    ax = axes[2]
    for a in rows:
        ax.plot(per_year["year"], per_year[a], "o-", color=SERIES[a], ms=3.5, label=SYM[a])
    ax.set_xticks(per_year["year"]); _grid(ax); ax.set_ylabel("$Q$"); ax.legend(fontsize=7.5, ncol=2)
    ax.set_title("(c) Test years", loc="left", fontsize=9)
    fig.tight_layout(); _save(fig, "fig_robust.png")


def fig_gsa(g, idx):
    names = {"grid_max": "grid limit", "c_unmet": "penalty", "battery_kwh": "battery", "dr_cap": "DR cap $\\rho^{\\max}$",
             "tariff_ratio": "tariff ratio"}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, o, title in ((axes[0], "gap_A1_A3", "(a) $Q(A_1)-Q(A_3)$"), (axes[1], "lead_A4", "(b) Lead of $A_4$")):
        sub = idx[idx.output == o]
        ax.scatter(sub.mu_star, sub.sigma, color=SERIES["A1"], s=18, zorder=3)
        tiny = sub[(sub.mu_star < 1e-6) & (sub.sigma < 1e-6)]
        for _, r in sub.drop(tiny.index).iterrows():
            ax.annotate(names[r.factor], (r.mu_star, r.sigma), fontsize=7.5, xytext=(3, 2), textcoords="offset points")
        if len(tiny):
            ax.annotate(", ".join(names[f] for f in tiny.factor) + " (no effect)", (0, 0), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points")
        _grid(ax, "both"); ax.set_xlabel(r"$\mu^\ast$"); ax.set_ylabel(r"$\sigma$"); ax.set_title(title, loc="left", fontsize=9)
        lim = max(sub.mu_star.max(), sub.sigma.max()) * 1.25
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax = axes[2]
    share = g["winner"].value_counts(normalize=True).reindex(["A1", "A2", "A3", "A4"]).fillna(0)
    ax.bar(range(4), share.values, color=[SERIES[a] for a in share.index], width=0.7, edgecolor="white")
    for x, v in enumerate(share.values):
        ax.text(x, v, f"{100*v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels([SYM[a] for a in share.index]); ax.set_ylim(0, 1.1); _grid(ax)
    ax.set_title("(c) Winner share", loc="left", fontsize=9)
    fig.tight_layout(); _save(fig, "fig_gsa.png")


def fig_oat(oat):
    params = list(dict.fromkeys(oat["parameter"]))
    n = len(params); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.4, 2.0 * nrow), sharey=True)
    axes = axes.ravel()
    for ax, par in zip(axes, params):
        g = oat[oat["parameter"] == par]; x = np.arange(len(g))
        for a in ["A1", "A2", "A3", "A4"]:
            ax.plot(x, g[a].to_numpy(float), color=SERIES[a], marker="o", ms=3.5, label=a)
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in g["value"]], fontsize=7.5)
        ax.set_title(par, loc="left", fontsize=8); _grid(ax); ax.axhline(0, color=INK2, lw=0.6, ls=":")
    for ax in axes[n:]:
        ax.axis("off")
    for r in range(nrow):
        axes[r * ncol].set_ylabel("$Q$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [NAMES[x] for x in l], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); _save(fig, "fig_oat.png")


def make_all(R, S, X, H, cells, risk, b_flat, Qs, per_year, oat, gsa_res, gsa_idx):
    fig_inputs(); fig_kpi(); fig_hfe(H); fig_scores(S); fig_drivers(); fig_coupling(cells, risk)
    fig_robust(b_flat, Qs, per_year); fig_gsa(gsa_res, gsa_idx); fig_oat(oat)


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
