"""
miner_2 novel factor exploration batch2 at 2031-08-11 (visible through 2031-08-08).
Focus on families with low library correlation. Gate: abs(IC)>=0.0070, abs(ICIR)>=0.0840.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr)

END = "2031-08-08"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

def full(name, f):
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10); cov = coverage_stats(f, fwd)
    corr, pairs = max_lib_corr(f, lib)
    return dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                covAD=cov["coverage_asset_days"], corr=corr)

cands = {}
# 1. Price-range position 20d: (close - low)/(high-low) sequential, -1 => short overbought
def range_pos(s, win=20):
    h = s.rolling(win).max(); l = s.rolling(win).min()
    return -((s - l) / (h - l))  # short overbought position
cands["range_pos_20"] = close.apply(lambda s: range_pos(s))

# 2. 1/up-move size ratio (upside semi-vol) flipped
up = ret.where(ret > 0, 0.0); dwn = ret.where(ret < 0, 0.0)
cands["up_down_vol_20"] = -(up.rolling(20).std() + 1e-12) / (dwn.rolling(20).std().abs() + 1e-12)

# 3. Conditional beta to VIX trend (defensive: sign of VIX change * beta)
vix_r = macro["VIX"].pct_change()
cov = ret.rolling(60).cov(vix_r); var = vix_r.rolling(60).var(); beta = cov.divide(var, axis=0)
vix_mom = macro["VIX"] / macro["VIX"].shift(20) - 1.0
cands["vix_beta_cond_60x20"] = beta.multiply(vix_mom, axis=0)  # want low beta when VIX up

# 4. conditional beta to USDCNY (China exosure) recent
cny_r = macro["USDCNY"].pct_change()
cov = ret.rolling(60).cov(cny_r); var = cny_r.rolling(60).var(); beta = cov.divide(var, axis=0)
cny_mom = macro["USDCNY"] / macro["USDCNY"].shift(20) - 1.0
cands["cny_beta_cond_60x20"] = -beta.multiply(cny_mom, axis=0)

# 5. 60d momentum (long-term trend)
cands["mom_60d"] = ret.rolling(60).sum()

# 6. Risk-parity weight proxy: -asset variance (prefer low vol)
cands["risk_parity_20"] = -ret.rolling(20).var()

# 7. Skewness 20d (flipped so high=negative skew? favor? just explore base)
cands["skew_20d"] = ret.rolling(20).skew()

# 8. Half-life trend: 5d ret / 20d vol (Sharpe)
cands["sharpe_5x20"] = ret.rolling(5).sum() / vol if (vol := ret.rolling(20).std()).notna().any() else ret.rolling(5).sum()

rows = []
for name, f in cands.items():
    rows.append(full(name, f))

print(f"\n{'factor':26s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} {'covAD':>6s} {'maxLib':>6s} {'gate':>4s}")
for r in rows:
    gp = abs(r["ic"]) >= 0.0070 and abs(r["icir"]) >= 0.0840
    print(f"{r['name']:26s} {r['ic']:7.4f} {r['icir']:7.3f} {r['hit']:5.2f} {r['n']:5d} "
          f"{r['covAD']:6.2f} {r['corr']:6.3f} {'PASS' if gp else ''}")