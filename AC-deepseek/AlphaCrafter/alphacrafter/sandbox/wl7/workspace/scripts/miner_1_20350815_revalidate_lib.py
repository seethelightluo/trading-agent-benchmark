"""miner_1 revalidation of active factor library as of 2035-08-14.

Re-checks the 8 persisted library factors on the full window plus recent
sub-windows (2y / 1y) against the admission gates:
  abs(daily paper IC)   >= 0.0070   at h=10
  abs(daily paper ICIR) >= 0.0840
Reports pairwise library signal correlation (max abs rho) for provenance.
"""
import importlib.util
import json
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("mshared", "scripts/miner_shared.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)

END = "2035-08-14"
GATE_IC = 0.0070
GATE_ICIR = 0.0840

close = ms.load_close(end=END)
macro = ms.load_macro(end=END)
ret = close.pct_change()

lib = {
    "rel_mom_20d_skip5": ms.lib_rel_mom(close, 20, 5),
    "beta_ew_60d": ms.lib_beta_ew(close, 60),
    "corr_ew_60": ms.lib_corr_ew(ret, 60),
    "downside_vol_ratio_20": ms.lib_downside_vol_ratio(close, 20),
    "kurt_20d_skip5": ms.lib_kurt(close, 20, 5),
    "max_ret_20d": ms.lib_max_ret(close, 20),
    "dxy_beta_cond_60x20": ms.lib_dxy_beta_cond(close, macro["DXY"], 60, 20),
    "eurusd_beta_cond_60x20": ms.lib_eurusd_beta_cond(close, macro["EURUSD"], 60, 20),
}

fwd10 = ms.forward_ret(close, 10)
fwd5 = ms.forward_ret(close, 5)
fwd20 = ms.forward_ret(close, 20)

cuts = {
    "full": close.index[0],
    "recent2y": pd.Timestamp("2033-08-15"),
    "recent1y": pd.Timestamp("2034-08-15"),
}

rows = []
for fid, fac in lib.items():
    for cname, cstart in cuts.items():
        mask = close.index >= cstart
        f = fac.loc[mask]
        c = close.loc[mask]
        f5 = ms.forward_ret(c, 5)
        f10 = ms.forward_ret(c, 10)
        f20 = ms.forward_ret(c, 20)
        ic5 = ms.ic_stats(ms.daily_ic(f, f5), 5)
        ic10 = ms.ic_stats(ms.daily_ic(f, f10), 10)
        ic20 = ms.ic_stats(ms.daily_ic(f, f20), 20)
        rows.append((fid, cname, ic10, ic5, ic20))

print(f"END={END}  gates: |IC|>={GATE_IC}, |ICIR|>={GATE_ICIR}")
print(f"{'factor':28s} {'window':9s} {'IC10':>8s} {'ICIR10':>8s} {'hit10':>6s} {'n':>5s} {'IC5':>8s} {'ICIR5':>7s} {'IC20':>8s} {'gate10':>6s}")
for fid, cname, ic10, ic5, ic20 in rows:
    gate = "PASS" if (abs(ic10["ic"]) >= GATE_IC and abs(ic10["icir"]) >= GATE_ICIR) else "FAIL"
    print(f"{fid:28s} {cname:9s} {ic10['ic']:8.4f} {ic10['icir']:8.3f} {ic10['hit']:6.2f} {ic10['n']:5d} "
          f"{ic5['ic']:8.4f} {ic5['icir']:7.3f} {ic20['ic']:8.4f} {gate:>6s}")

# ---- pairwise library signal correlation (artifact-level, h=10-aligned fwd basis) ----
print("\n--- pairwise library correlation (max abs rho) ---")
facs = {k: v for k, v in lib.items()}
names = list(facs.keys())
corr_tab = pd.DataFrame(index=names, columns=names, dtype=float)
for i in names:
    vi = facs[i].stack()
    for j in names:
        vj = facs[j].stack()
        cc = pd.concat([vi, vj], axis=1, join="inner").dropna()
        corr_tab.loc[i, j] = np.corrcoef(cc.iloc[:, 0], cc.iloc[:, 1])[0, 1] if len(cc) > 30 else np.nan
print(corr_tab.round(3).to_string())
maxrho = 0.0
maxpair = None
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r = abs(corr_tab.iloc[i, j])
        if not np.isnan(r) and r > maxrho:
            maxrho, maxpair = r, (names[i], names[j])
print(f"\nmax_abs_library_correlation = {maxrho:.3f}  pair={maxpair}")