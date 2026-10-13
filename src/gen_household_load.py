"""Generate the stochastic neighbourhood load (baseline load model).

14 apartments (household sizes 5x1, 5x2, 2x3, 1x4, 1x5 persons; mean 2.14, close to
the Swiss average) are simulated with richardsonpy (Richardson et al., 2010) at 1-min
resolution for every calendar year 2015-2021. Lighting is driven by the measured Zurich
irradiance. Electric space-heating appliances are excluded (prev_heat_dev=True) and each
apartment is normalised to the Stromspiegel reference demand for apartments
(is_sfh=False, do_normalization=True). Every (household, year) uses its own seed, so each
year is an independent realisation of the neighbourhood. The 15-min sum is written to
data/household_load_15min.csv (kW); ems_data.load_series() rescales it to the common
annual demand.

    python src/gen_household_load.py        # ~10-20 min
"""
import os, sys, random, warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ems_data  # noqa: E402

SIZES = [1] * 5 + [2] * 5 + [3] * 2 + [4, 5]
SEED = 2026


def one(args):
    hh, occ_n, year, direct, diffuse = args
    import richardsonpy.classes.occupancy as occ
    import richardsonpy.classes.electric_load as eload
    seed = SEED * 1000 + hh * 10 + (year - 2000) * 1_000_000
    random.seed(seed); np.random.seed(seed % (2 ** 32))
    first = pd.Timestamp(f"{year}-01-01").dayofweek + 1          # 1 = Monday
    ndays = 366 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 365
    o = occ.Occupancy(number_occupants=occ_n, initial_day=first, nb_days=ndays)
    e = eload.ElectricLoad(occ_profile=o.occupancy, total_nb_occ=occ_n, q_direct=direct, q_diffuse=diffuse,
                           is_sfh=False, prev_heat_dev=True, timestep=60, initial_day=first,
                           do_normalization=True)
    lc = np.asarray(e.loadcurve, float)[: ndays * 1440] / 1000.0     # kW, 1-min
    return year, hh, lc.reshape(-1, 15).mean(axis=1)


def main():
    raw = ems_data.zurich_raw()
    comp = ems_data.solar_components(raw)
    jobs = []
    for y in ems_data.YEARS:
        sel = comp.index.year == y
        direct = np.repeat(comp.loc[sel, "direct_h"].to_numpy(), 15)
        diffuse = np.repeat(comp.loc[sel, "dhi"].to_numpy(), 15)
        for hh, n in enumerate(SIZES):
            jobs.append((hh, n, y, direct, diffuse))
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        res = pool.map(one, jobs, chunksize=1)
    parts = []
    for y in ems_data.YEARS:
        tot = sum(r[2] for r in res if r[0] == y)
        idx = pd.date_range(f"{y}-01-01", periods=len(tot), freq="15min")
        parts.append(pd.Series(tot, index=idx))
    s = pd.concat(parts); s.index.name = "Timestamp"
    s.rename("load_kw").to_frame().round(5).to_csv(ems_data.HOUSEHOLD_CSV)
    per_hh = {(r[0], r[1]): r[2].sum() * 0.25 for r in res}
    print("household-years:", len(res), " mean kWh/household-year:", round(np.mean(list(per_hh.values())), 1))
    print("neighbourhood kWh by year:", (s.groupby(s.index.year).sum() * 0.25).round(0).to_dict())


if __name__ == "__main__":
    main()
