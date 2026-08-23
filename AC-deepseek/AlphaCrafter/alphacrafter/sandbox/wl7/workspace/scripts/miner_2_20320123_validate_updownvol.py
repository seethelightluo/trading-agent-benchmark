"""miner_2 validation of updownvol_60 at 2032-01-22 (visible through last completed trading day).
Gate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 @ h10. Full decay + orthogonality + regime splits."""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover)

END = "2032-01-22"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change()
ep = 1e-9
up = ret.where(ret > 0, 0.0); dwn = ret.where(ret < 0, 0.0)
factor = -(up.rolling(60).std()+ep)/(dwn.abs().rolling(60).std()+ep)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

# decay across horizons
print("\n[decay]")
decay = {}
for h in [1,2,3,5,8,10,15,20]:
    fwd = forward_ret(close, h)
    ic = daily_ic(factor, fwd)
    st = ic_stats(ic, h)
    decay[h] = st
    print(f"  h={h:>3d}  ic={st['ic']:+.4f}  icir={st['icir']:+.4f}  hit={st['hit']:.3f}  n={st['n']}")

h10 = decay[10]
print("\n[regime splits]")
fwd10 = forward_ret(close, 10)
ic_all = daily_ic(factor, fwd10)
ic_all_s = ic_all.dropna()
# recent 6 months (~126 trading days) and last quarter (~63)
for label, sl in [("recent_6m", ic_all_s.index >= "2031-08-01"),
                  ("recent_3m", ic_all_s.index >= "2031-11-01"),
                  ("prev_6m", (ic_all_s.index >= "2031-02-01") & (ic_all_s.index < "2031-08-01"))]:
    sub = ic_all_s[sl]
    icm = sub.mean(); icr = icm/sub.std(ddof=1) if sub.std(ddof=1)>0 else np.nan
    print(f"  {label}: ic={icm:+.4f} icir={icr:+.4f} hit={(sub>0).mean():.3f} n={len(sub)}")

# coverage, turnover
cov = coverage_stats(factor, fwd10)
turn = rank_turnover(factor, 10)
print(f"\n[coverage] {cov}")
print(f"[turnover] rank_turnover(10)={turn:.3f}")

# orthogonality
corr, pairs = max_lib_corr(factor, lib)
print(f"\n[max_lib_corr] {corr:.4f}")
print("pairs:", json.dumps(pairs))

# library correlation of candidate vs the two most similar (downside_vol_ratio, kurt, rev)
flat = factor.stack()
for nm in lib:
    p = lib[nm].reindex(factor.index).stack()
    df = pd.concat([flat.rename("f"), p.rename("p")], axis=1).dropna()
    rho = df["f"].corr(df["p"]) if len(df)>30 else np.nan
    print(f"  corr vs {nm}: {rho:+.4f}")

ok = abs(h10["ic"]) >= 0.0070 and abs(h10["icir"]) >= 0.0840
print("\n[GATE] abs_ic>=0.0070 and abs_icir>=0.0840:", ok)