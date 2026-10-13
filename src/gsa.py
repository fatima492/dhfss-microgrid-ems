"""Morris elementary-effects screening (Morris, 1991; Campolongo et al., 2007) with SALib.

Each sample re-tunes the rule thresholds on the training years and re-simulates the test
years. Outputs: gap between the two leading baseline policies, Q(A1) - Q(A3), and the lead
of the hybrid policy, Q(A4) - max_{i != 4} Q(A_i); the winner is recorded as well.
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

FACTORS = [("grid_max", 8.0, 20.0), ("c_unmet", 1.0, 10.0), ("battery_kwh", 10.0, 40.0),
           ("dr_cap", 0.05, 0.25), ("tariff_ratio", 1.5, 4.0)]
OUTPUTS = ["gap_A1_A3", "lead_A4"]
R_TRAJ = 20
LEVELS = 4


def _eval(x):
    import warnings; warnings.simplefilter("ignore")
    import ems_data as E, mcdm as D, pipeline as P
    params = {f[0]: float(v) for f, v in zip(FACTORS[:4], x[:4])}
    prices = dict(E.CONFIG["prices"]); prices["t3_onpeak"] = 0.10 * float(x[4])
    R = P.run(params, prices=prices)
    Q, _, _, _ = P.base_Q(R["blocks"])
    return {**{f[0]: float(v) for f, v in zip(FACTORS, x)}, "Q_A1": Q[0], "Q_A2": Q[1], "Q_A3": Q[2], "Q_A4": Q[3],
            "gap_A1_A3": Q[0] - Q[2], "lead_A4": Q[3] - max(Q[0], Q[1], Q[2]),
            "winner": P.ALTS[int(np.argmax(Q))], "targets": f"{R['targets']['A2']}/{R['targets']['A4']}"}


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
