"""miner_3 replication validation of library factors as-of 2035-11-15."""
import sys, os, json, math
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location("m3lib", "scripts/miner3_lib.py")
m3lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3lib)

os.chdir(os.path.dirname(os.path.abspath(__file__))) if False else None
C, V, H, L, O = m3lib.load_close_panel(4000)
R = C.pct_change()

rets = {s: C[s].pct_change() for s in m3lib.WATCHLIST}
R = pd.DataFrame(rets).sort_index()

mkt = R[m3lib.WATCHLIST].mean(axis=1)
mkt_ret_20 = mkt.rolling(20).mean()
mkt_ret_60 = mkt.rolling(60).mean()
mkt_vol_20 = mkt.rolling(20).std()
mkt_vol_60 = mkt.rolling(60).std()

def tuw(C, s, n=120):
    rl = C[s] / C[s].rolling(n).max() - 1.0
    return rl * -1.0  # deeper dd -> larger value

def r2_signed(C, s, n=30):
    y = C[s].iloc[-n:].values
    x = np.arange(len(y), dtype=float)
    if np.std(y) < 1e-12:
        return np.nan
    r = np.corrcoef(x, y)[0, 1]
    return r

mkt_col = 'mkt'
R2 = R.copy()

# candidate library factors
def semi_down_ratio(R, w=20):
    r = R.rolling(w).apply(lambda x: (x<0).sum()/max(len(x),1), raw=True)
    return r

panels = {}
panels['semi_down_ratio_20'] = semi_down_ratio(R, 20)
panels['mom_120d_skip5'] = C / C.shift(125) - 1.0
panels['mom_10d_skip5'] = C / C.shift(15) - 1.0
panels['trend_r2_30_signed'] = pd.DataFrame({s: C[s].rolling(30).apply(lambda y: np.corrcoef(np.arange(len(y)), y)[0,1] if np.std(y)>1e-12 else np.nan, raw=True) * C[s].pct_change(30).apply(lambda x: 1 if x>0 else -1 if x<0 else 0) for s in C.columns}).sort_index()

for fid, panel in panels.items():
    s = m3lib.rank_ic(panel, R.shift(-10))
    summ = m3lib.summarize(s, 10, fid)
    print(fid, "IC", summ['ic'], "ICIR", summ['icir'], "hit", summ['ic_hit_ratio'], "n", summ['n_ic_dates'])
    print("  regime:", summ['regime'])