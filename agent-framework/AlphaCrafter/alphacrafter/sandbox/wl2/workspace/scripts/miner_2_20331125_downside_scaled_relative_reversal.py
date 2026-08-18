import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index()
r=P.pct_change()
# Downside-risk-adjusted relative reversal: recent 5d dislocation, scaled by
# 20d downside deviation; explicit min observations and one-day information lag.
down=r.where(r<0,0.0)
dv=down.rolling(20,min_periods=15).apply(lambda x: np.sqrt(np.mean(np.asarray(x)**2)),raw=True)
raw=-(P.pct_change(5)-P.pct_change(5).median(axis=1,skipna=True).values[:,None])/dv
f=raw.shift(1)
for h in [5,10,20]:
  vals=[]; ns=[]
  y=P.shift(-h)/P-1
  for d in P.index:
    ok=f.loc[d].notna()&y.loc[d].notna()
    if ok.sum()>=8:
      vals.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic); ns.append(ok.sum())
  z=np.asarray(vals)
  print('horizon',h,'dates',len(z),'avg_n',round(np.mean(ns),2) if ns else 0,'coverage',round(np.mean(ns)/15,4) if ns else 0,'IC',round(z.mean(),6) if len(z) else 'nan','ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else 'nan','hit',round((z>0).mean(),4) if len(z) else 'nan')
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_2_20331125_downside_scaled_relative_reversal_signal.csv')
