"""miner_1 2027-02-19: quick screen of candidate trend/momentum-family factors on 15-instrument cross-asset panel.
Screening pass only - each promising candidate gets its own dedicated validation script afterwards."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner1_common import ic_analysis, coverage, turnover

with open('scripts/panel_cache.pkl', 'rb') as f:
    P = pd.read_pickle(f)
close = P['close']; high = P['high']; low = P['low']; ret = P['ret']
START = '2021-01-01'

def zscore(s, window=120):
    return (s - s.rolling(window, min_periods=20).mean()) / s.rolling(window, min_periods=20).std()

def candidate(name, fn):
    fdf = pd.DataFrame({s: fn(close[s], high[s], low[s]) for s in close.columns})
    fdf = fdf[fdf.index >= START]
    c = coverage(fdf, close)
    t = turnover(fdf)
    ic1 = ic_analysis(fdf, close, fwd_days=1)
    ic5 = ic_analysis(fdf, close, fwd_days=5)
    ic10 = ic_analysis(fdf, close, fwd_days=10)
    print(f"=== {name} ===")
    print(f"  coverage={c:.3f} turnover10d={t:.3f}")
    for tag, r in [('1d', ic1), ('5d', ic5), ('10d', ic10)]:
        print(f"  fwd{tag:>2}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n_dates={r['n_dates']} n_obs={r['n_obs']}")
    return {'name': name, 'cov': c, 'turn': t, 'ic1': ic1['ic'], 'icir1': ic1['icir'],
            'ic5': ic5['ic'], 'icir5': ic5['icir'], 'ic10': ic10['ic'], 'icir10': ic10['icir']}

results = []

# A: 60d momentum (skip 5) / 20d realized vol - risk-adjusted trend
def fa(c, h, l):
    mom = c.shift(5) / c.shift(65) - 1.0
    vol = ret[c.name].rolling(20).std()
    return mom / vol
results.append(candidate('vol_adj_mom_60d_skip5', fa))

# B: trend slope t-stat over 60d (OLS slope / se of log price)
def fb(c, h, l):
    lp = np.log(c)
    x = np.arange(60)
    def tstat(ts):
        if len(ts) < 60 or np.isnan(ts).any():
            return np.nan
        y = ts.values
        b = np.polyfit(x, y, 1)[0]
        resid = y - np.polyval(np.polyfit(x, y, 1), x)
        se = np.std(resid) / np.sqrt(np.sum((x - x.mean()) ** 2))
        return b / se if se > 0 else np.nan
    return lp.rolling(60).apply(tstat, raw=False)
results.append(candidate('trend_slope_tstat_60d', fb))

# C: multi-horizon momentum composite (z-scored 20/60/120d momentum, skip 5)
def fc(c, h, l):
    m20 = c.shift(5) / c.shift(25) - 1.0
    m60 = c.shift(5) / c.shift(65) - 1.0
    m120 = c.shift(5) / c.shift(125) - 1.0
    return (zscore(m20) + zscore(m60) + zscore(m120)) / 3.0
results.append(candidate('mom_multi_horizon_20_60_120', fc))

# D: position within 60d high-low range (0..1)
def fd(c, h, l):
    hh = h.rolling(60).max(); ll = l.rolling(60).min()
    return (c - ll) / (hh - ll).replace(0, np.nan)
results.append(candidate('hl_pos_60d', fd))

# E: commodity-complex relative momentum (60d skip5 minus cross-sectional median of XAU/COPPER/WTI)
def fe(c, h, l):
    mom = c.shift(5) / c.shift(65) - 1.0
    comm = close[['XAU', 'COPPER', 'WTI']]
    comm_mom = comm.shift(5) / comm.shift(65) - 1.0
    med = comm_mom.median(axis=1)
    return mom - med.reindex(mom.index)
results.append(candidate('comm_rel_mom_60d_skip5', fe))

# F: 20d momentum skip 5 / 60d vol (shorter risk-adjusted trend)
def ff(c, h, l):
    mom = c.shift(5) / c.shift(25) - 1.0
    vol = ret[c.name].rolling(60).std()
    return mom / vol
results.append(candidate('vol_adj_mom_20d_skip5', ff))

print("\n=== SUMMARY ===")
for r in results:
    print(f"{r['name']:32s} cov={r['cov']:.3f} turn={r['turn']:.3f} "
          f"IC1={r['ic1']:+.4f}/{r['icir1']:+.3f} IC5={r['ic5']:+.4f}/{r['icir5']:+.3f} IC10={r['ic10']:+.4f}/{r['icir10']:+.3f}")
