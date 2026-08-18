import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
    d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).sort_values('date')
    px[a]=d.set_index('date')['close']
close=pd.DataFrame(px).sort_index().ffill()
r=close.pct_change()
# short-term reversal weighted by downside risk: negative recent return, penalize assets with high downside vol
sig=-(close/close.shift(10)-1)/(r.where(r<0,0).rolling(20,min_periods=15).std()*np.sqrt(252)+1e-8)
# prevent lookahead: signal at t uses through t-1
sig=sig.shift(1)
fwd=close.shift(-10)/close-1
ics=[]; ns=[]; turnovers=[]
prev=None
for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ics.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
        ranks=x[ok].rank(pct=True)
        if prev is not None: turnovers.append(np.mean(abs(ranks-prev.reindex(ranks.index).fillna(.5))))
        prev=ranks
z=np.array(ics); ic=z.mean(); ir=ic/(z.std(ddof=1)+1e-12)*np.sqrt(252)
print(f'dates={len(z)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(z>0):.4f} coverage={np.mean(ns)/15:.4f} turnover={np.mean(turnovers):.4f}')
for n in [260,520,780]:
 q=z[-n:]; print(f'recent{n}: IC={q.mean():.6f} ICIR={q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(252):.6f} N={len(q)}')
# horizons decay with same signal
for h in [1,5,10,20,30]:
 y=close.shift(-h)/close-1; zz=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: zz.append(spearmanr(sig.loc[dt][ok],y.loc[dt][ok]).statistic)
 print(f'h{h}: IC={np.mean(zz):.6f} n={len(zz)}')
