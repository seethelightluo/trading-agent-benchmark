"""miner_2 exploration of new candidate factors through visible 2034-08-15."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import miner_shared as M

END = "2034-08-15"
H = 10
close = M.load_close(END)
macro = M.load_macro(END)
fwd = M.forward_ret(close, H)
lib = M.library_panel(close, macro)
ret = close.pct_change()

def ic_stats_of(panel, direction=1):
    ic = M.daily_ic(panel, fwd)
    st = M.ic_stats(ic, H)
    adj = ic * direction
    icir = adj.mean()/adj.std(ddof=1) if adj.std(ddof=1)>0 else np.nan
    gate = (abs(st["ic"]) >= M.IC_GATE) and (abs(icir) >= M.ICIR_GATE)
    best, pairs = M.max_lib_corr(panel, lib)
    return dict(ic=st["ic"], icir=float(icir), hit=st["hit"], n=st["n"],
                gate=bool(gate), maxrho=best)

cands = {}

# 1. USDJPY-beta conditional 60x20 (novel macro signal, not in library)
fx = macro["USDJPY"]
fx_r = fx.pct_change()
cov = ret.rolling(60, min_periods=30).cov(fx_r)
var = fx_r.rolling(60, min_periods=30).var()
beta = cov.divide(var, axis=0)
fx_mom = fx / fx.shift(20) - 1.0
cands["usdjpy_beta_cond_60x20"] = beta.multiply(fx_mom, axis=0)

# 2. VIX-beta conditional 60x20 (defensive)
vix = macro["VIX"]; vix_r = vix.pct_change()
cov = ret.rolling(60, min_periods=30).cov(vix_r)
var = vix_r.rolling(60, min_periods=30).var()
beta = cov.divide(var, axis=0)
vix_mom = vix / vix.shift(20) - 1.0
cands["vix_beta_cond_60x20"] = -beta.multiply(vix_mom, axis=0)

for name, panel in cands.items():
    st = ic_stats_of(panel)
    print(f"\n{name}: IC {st['ic']:+.4f} ICIR {st['icir']:+.3f} hit {st['hit']:.3f} "
          f"n {st['n']} gate={st['gate']} maxrho {st['maxrho']:.3f}")

out = {k: ic_stats_of(v) for k, v in cands.items()}
json.dump(dict(end=END, horizon=H, results=out),
          open("scripts/miner2_20340816_explore.json", "w"), indent=1, default=str)
print("\nsaved explore json")
