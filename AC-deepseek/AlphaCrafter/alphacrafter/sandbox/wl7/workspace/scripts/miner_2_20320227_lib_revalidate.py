"""miner_2 revalidation of active library factors at 2032-02-27 (visible through 2032-02-26).
Gate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 @ h10. Reports full-recent-quarter (drift) splits.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover)

END = "2032-02-26"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

rows = []
for name, panel in lib.items():
    f = panel
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10)
    ic_s = ic.dropna()
    # recent splits
    def split(mask):
        sub = ic_s[mask]
        m = sub.mean(); sd = sub.std(ddof=1) if sub.std(ddof=1) > 0 else np.nan
        return m, m / sd if np.isfinite(sd) else np.nan
    ic_r, icir_r = split(ic_s.index >= "2031-11-01")
    ic_q, icir_q = split(ic_s.index >= "2032-02-01")
    cov = coverage_stats(f, fwd); turn = rank_turnover(f, 10)
    rows.append(dict(name=name, ic=round(st["ic"], 4), icir=round(st["icir"], 3),
                     hit=round(st["hit"], 3), n=st["n"],
                     ic_r=round(ic_r, 4), icir_r=round(icir_r, 3),
                     ic_q=round(ic_q, 4), icir_q=round(icir_q, 3),
                     covAD=round(cov["coverage_asset_days"], 3), turn=round(turn, 3)))

for r in rows:
    flag = "PASS" if abs(r["ic"]) >= 0.007 and abs(r["icir"]) >= 0.084 else "fail"
    print(f"{r['name']:26s} ic={r['ic']:+.4f} icir={r['icir']:+.4f} hit={r['hit']:.3f} "
          f"n={r['n']} | recent_3m ic={r['ic_r']:+.4f} icir={r['icir_r']:+.3f} "
          f"| Q(Feb) ic={r['ic_q']:+.4f} icir={r['icir_q']:+.3f} covAD={r['covAD']} turn={r['turn']} [{flag}]")

json.dump(rows, open("scripts/miner_2_20320227_lib_revalidate.json", "w"), indent=1)
print("\nsaved scripts/miner_2_20320227_lib_revalidate.json")