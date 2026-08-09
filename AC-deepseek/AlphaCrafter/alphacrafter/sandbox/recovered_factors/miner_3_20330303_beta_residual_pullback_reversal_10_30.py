"""One idea: medium-horizon beta-residual pullback reversal (10d/30d).
Higher score identifies an asset whose 10-day idiosyncratic return has been
unusually negative after removing its rolling exposure to the equal-weight
cross-asset return.  This is a lower-turnover extension of the prior 5d signal.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CUT = pd.Timestamp('2033-03-02')  # last completed bar available on 2033-03-03
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(asset):
    return (pd.read_csv('../persistent/stock_data/' + asset + '.csv', parse_dates=['date'])
            .set_index('date')['close'].rename(asset))

p = pd.concat([load(a) for a in ASSETS], axis=1).sort_index().loc[:CUT]
r = p.pct_change()
m = r.mean(axis=1)
beta = r.apply(lambda x: x.rolling(60, min_periods=42).cov(m)).div(
    m.rolling(60, min_periods=42).var() + 1e-12, axis=0)
resid = r - beta.mul(m, axis=0)
# One factor: a standardized 10-day residual pullback, high values imply rebound.
f = -resid.rolling(10, min_periods=7).sum().div(
    resid.rolling(30, min_periods=21).std() * np.sqrt(10) + 1e-12)

print('CANDIDATE beta_residual_pullback_reversal_10_30 cutoff', CUT.date(),
      'calendar_dates', len(p), 'assets', len(ASSETS))
print('valid_dates', int(f.notna().any(axis=1).sum()),
      'coverage', round(float(f.notna().mean().mean()), 6),
      'valid_cells', int(f.notna().sum().sum()))
ics = {}
for h in (1, 3, 5, 7, 10, 20):
    fw = p.shift(-h).div(p) - 1
    vals, ns = [], []
    for d in f.index:
        q = pd.concat([f.loc[d].rename('f'), fw.loc[d].rename('y')], axis=1).dropna()
        if len(q) >= 8 and q.f.nunique() > 1:
            z = spearmanr(q.f, q.y).statistic
            if np.isfinite(z): vals.append((d, z)); ns.append(len(q))
    x = pd.Series(dict(vals), dtype=float); ics[h] = x
    sd = x.std(ddof=1)
    print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f' %
          (h, x.mean(), x.mean()/sd, len(x), (x > 0).mean(), np.mean(ns)))
    if h == 10:
        for name, lo, hi in [('2020-2024','2020-01-01','2024-12-31'),
                             ('2025-2026','2025-01-01','2026-12-31'),
                             ('2027+','2027-01-01',str(CUT.date()))]:
            z = x.loc[lo:hi]
            print('REGIME10', name, 'dates', len(z), 'IC', round(z.mean(),6),
                  'ICIR', round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,
                  'hit', round((z>0).mean(),4))
ranks = f.rank(axis=1, pct=True)
turn = []
for j in range(1, len(ranks)):
    q = ranks.iloc[[j-1,j]].T.dropna()
    if len(q) >= 8:
        turn.append(1-spearmanr(q.iloc[:,0], q.iloc[:,1]).statistic)
print('RANK_TURNOVER', round(float(np.mean(turn)),6), 'pairs',len(turn))
print('DECAY', {h:(round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)),6),len(x)) for h,x in ics.items()})
f.to_pickle('scripts/miner_3_20330303_beta_residual_pullback_reversal_10_30_signal.pkl')
