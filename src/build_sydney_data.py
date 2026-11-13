"""Build the second-site input file (Sydney, Australia) from the Ausgrid "Solar home electricity" data.

Source: Ausgrid solar home electricity data, 1 July 2012 - 30 June 2013 (file
"2012-2013 Solar home electricity data v2.csv", 300 residential customers with rooftop PV,
30-min metered general consumption GC, controlled load CL and gross PV generation GG), described by
Ratnam et al. (2017), Int. J. Sustainable Energy 36(8):787-806, doi:10.1080/14786451.2015.1100196.
The original download page at ausgrid.com.au is no longer online; the file used here is the copy in
Data.zip of https://github.com/rockey-abhishek/Ausgrid-Dataset (MD5 of the CSV given below).

Construction (all choices fixed in advance, seed 2026):
* customers with a complete year of GC and GG records and a plausible PV yield
  (800-1,800 kWh/kWp/yr) and general consumption (1.5-15 MWh/yr) are retained;
* household load: sum of the general consumption (GC) of 14 customers drawn at random from the
  retained set (controlled off-peak water heating, CL, is excluded, as electric space and water
  heating are excluded from the Zurich load model);
* PV: median over all retained customers of the measured output per kWp (robust to individual
  shading and outages), later scaled to the 20 kWp array of the case study;
* 30-min values are assigned to both 15-min steps of the interval they end (energy preserving).
Times are the local times of the published data.

Usage:  python src/build_sydney_data.py "path/to/2012-2013 Solar home electricity data v2.csv"
Writes: data/sydney_ausgrid_15min.csv
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd

RAW_MD5 = "f2aaa003772dbd4329935a525066d733"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sydney_ausgrid_15min.csv")
N_HOMES = 14
SEED = 2026


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(raw_path):
    if md5(raw_path) != RAW_MD5:
        raise RuntimeError("Ausgrid file does not match the expected checksum")
    raw = pd.read_csv(raw_path, skiprows=1)
    tcols = [c for c in raw.columns if ":" in c]
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y")
    days = pd.date_range("2012-07-01", "2013-06-30", freq="D")
    cap = raw.groupby("Customer")["Generator Capacity"].first()
    def wide(cat):
        sub = raw[raw["Consumption Category"] == cat]
        return {c: g.set_index("date")[tcols].reindex(days) for c, g in sub.groupby("Customer")}
    gc, gg = wide("GC"), wide("GG")
    keep = []
    for c in sorted(set(gc) & set(gg)):
        a, b = gc[c], gg[c]
        if a.isna().any().any() or b.isna().any().any():
            continue
        yield_kwh_kwp = b.to_numpy().sum() / cap[c]
        load_mwh = a.to_numpy().sum() / 1000
        if 800 <= yield_kwh_kwp <= 1800 and 1.5 <= load_mwh <= 15:
            keep.append(c)
    rng = np.random.default_rng(SEED)
    homes = sorted(rng.choice(keep, size=N_HOMES, replace=False).tolist())
    # 30-min interval ending at the column label; 24:00 / 0:00 handled as end of day
    idx30 = pd.DatetimeIndex([d + pd.Timedelta(minutes=30 * (k)) for d in days for k in range(48)])  # interval starts
    load30 = sum(gc[c].to_numpy().ravel() for c in homes) * 2.0               # kWh per 30 min -> kW
    cf30 = np.median(np.stack([gg[c].to_numpy().ravel() * 2.0 / cap[c] for c in keep]), axis=0)  # kW per kWp
    s = pd.DataFrame({"load_kw": load30, "pv_kw_per_kwp": cf30}, index=idx30)
    idx15 = pd.date_range(idx30[0], idx30[-1] + pd.Timedelta(minutes=15), freq="15min")
    out = s.reindex(idx15).ffill()
    out.index.name = "Timestamp"
    out.attrs = {}
    out.to_csv(OUT, float_format="%.5f")
    with open(OUT.replace(".csv", "_homes.txt"), "w") as fh:
        fh.write(f"retained customers: {len(keep)}\nselected (seed {SEED}): {homes}\n")
    print(f"retained {len(keep)} customers; homes {homes}; annual load {load30.sum()/2:.0f} kWh; "
          f"PV yield {cf30.sum()/2:.0f} kWh/kWp")


if __name__ == "__main__":
    main(sys.argv[1])
