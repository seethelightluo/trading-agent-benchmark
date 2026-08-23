"""miner_2 factor exploration: vix-regime-conditional momentum/reversal switch."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import miner_shared as M

END = "2033-05-17"
H = 10
close = M.load_close(END)
macro = M.load_macro(END)
vix = macro["VIX"]
vix_pctile = vix.rolling(90, min_periods=30).rank(pct=True)  # trailing 90d percentile rank

mom_pre = close / close.shift(25) - 1.0
mom = mom_pre.subtract(mom_pre.median(axis=1), axis=0)

# soft switch: 1.0 when VIX percentile high, 0.0 when low
w = vix_pctile  # between 0 and 1
# conditional: high VIX -> reversal (negative), low VIX -> momentum
factor = (1.0 - w) * mom - w * mom
factor = factor - factor.mean(axis=1)

print(f"=== vix_regime_switch END={END} horizon={H} ===")
print("dates:", close.shape[0], "assets:", close.shape[1])
fwd = M.forward_ret(close, H)
ic = M.daily_ic(factor, fwd)
st = M.ic_stats(ic, H)
cov = M.coverage_stats(factor, fwd)
turn = M.rank_turnover(factor)
gate = (abs(st["ic"]) >= M.IC_GATE) and (abs(st["icir"]) >= M.ICIR_GATE)
s = ic.dropna()
print(f"IC {st['ic']:+.4f} ICIR {st['icir']:+.3f} hit {st['hit']:.3f} n {st['n']} gate={gate}")
print(f"covAD {cov['coverage_asset_days']:.3f} dates_ge8 {cov['coverage_dates_ge8']:.3f} turn {turn:.2f}")
py = {}
for y, g in s.groupby(s.index.year):
    m = g.mean(); sd = g.std(ddof=1)
    py[int(y)] = round(float(m), 4)
print("per_year:", py)
for lab, dt in [("r6m", "2032-11-17"), ("r1y", "2032-05-17")]:
    r = s[s.index >= dt]
    if len(r):
        m = r.mean(); sd = r.std(ddof=1)
        print(f"{lab} IC {m:+.4f} ICIR {m/sd if sd>0 else float('nan'):.3f} hit {(r>0).mean():.3f}")

lib = M.library_panel(close, macro)
best, pairs = M.max_lib_corr(factor, lib)
print("max_lib_corr:", best, pairs)
print("gate:", gate)

# decay across horizons
print("decay (IC by horizon):")
for h in (3, 5, 10, 20):
    fh = M.forward_ret(close, h)
    ih = M.daily_ic(factor, fh)
    sh = M.ic_stats(ih, h)
    print(f"  h={h}: IC {sh['ic']:+.4f} ICIR {sh['icir']:+.3f}")