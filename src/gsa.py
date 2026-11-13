"""Morris elementary-effects screening (Morris, 1991; Campolongo et al., 2007) with SALib, and the
two-parameter preference map of the revision.

Each sample re-tunes the rule settings on the training years (cost objective for A2/A4,
multi-criteria objective for A5/A6) and re-simulates the test years. Outputs (revision):
* gap_A5_A6: Q(A5) - Q(A6), the two multi-criteria-tuned storage policies (baseline leaders);
* lead_storage: max(Q(A5), Q(A6)) - max(Q(A1), Q(A3)), the lead of the best tuned storage policy
  over the best policy without storage; the winner is recorded as well.
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

FACTORS = [("grid_max", 8.0, 20.0), ("c_unmet", 1.0, 10.0), ("battery_kwh", 10.0, 40.0),
           ("dr_cap", 0.05, 0.25), ("tariff_ratio", 1.5, 4.0)]
OUTPUTS = ["gap_A5_A6", "lead_storage"]
R_TRAJ = 20
LEVELS = 4


def _targets_str(R, P):
    return "; ".join(f"{a}:{P.setting_label(s)}" for a, s in R["targets"].items())


def _eval(x):
    import warnings; warnings.simplefilter("ignore")
    import ems_data as E, pipeline as P
    params = {f[0]: float(v) for f, v in zip(FACTORS[:4], x[:4])}
    prices = dict(E.CONFIG["prices"]); prices["t3_onpeak"] = 0.10 * float(x[4])
    R = P.run(params, prices=prices)
    Q, _, _, _ = P.base_Q(R["blocks"])
    q = dict(zip(P.ALTS, Q))
    return {**{f[0]: float(v) for f, v in zip(FACTORS, x)}, **{f"Q_{a}": q[a] for a in P.ALTS},
            "gap_A5_A6": q["A5"] - q["A6"], "lead_storage": max(q["A5"], q["A6"]) - max(q["A1"], q["A3"]),
            "winner": P.ALTS[int(np.argmax(Q))], "targets": _targets_str(R, P)}


def preference_eval(args):
    """Winner, runner-up and margin for one (grid limit, tariff ratio) cell, with re-tuning."""
    import warnings; warnings.simplefilter("ignore")
    import ems_data as E, pipeline as P
    gmax, ratio = args
    prices = dict(E.CONFIG["prices"]); prices["t3_onpeak"] = 0.10 * ratio
    R = P.run({"grid_max": gmax}, prices=prices)
    Q, _, _, _ = P.base_Q(R["blocks"])
    o = np.argsort(-Q)
    return {"grid_max": gmax, "tariff_ratio": ratio, **dict(zip(P.ALTS, Q)), "winner": P.ALTS[o[0]],
            "runner_up": P.ALTS[o[1]], "margin": float(Q[o[0]] - Q[o[1]]), "targets": _targets_str(R, P)}


def problem():
    return {"num_vars": len(FACTORS), "names": [f[0] for f in FACTORS], "bounds": [[f[1], f[2]] for f in FACTORS]}


def run_morris(path, seed=2026):
    from SALib.sample import morris as ms
    X = ms.sample(problem(), N=R_TRAJ, num_levels=LEVELS, seed=seed)
    partial = path + ".partial.csv"
    done = pd.read_csv(partial) if os.path.exists(partial) else pd.DataFrame()
    start = len(done); rows = done.to_dict(orient="records")
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        for k, row in enumerate(pool.imap(_eval, list(X[start:]), chunksize=1), start=start + 1):
            rows.append(row)
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f"GSA sample {k}/{len(X)}", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    os.remove(partial)


def indices(g):
    from SALib.analyze import morris as ma
    X = g[[f[0] for f in FACTORS]].to_numpy()
    out = []
    for o in OUTPUTS:
        res = ma.analyze(problem(), X, g[o].to_numpy(), num_levels=LEVELS, seed=1)
        for k, f in enumerate(FACTORS):
            out.append({"output": o, "factor": f[0], "mu_star": res["mu_star"][k], "sigma": res["sigma"][k],
                        "mu": res["mu"][k], "mu_star_conf": res["mu_star_conf"][k]})
    return pd.DataFrame(out)
