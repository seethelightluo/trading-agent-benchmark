import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-02-24')
def rd(path): return pd.read_csv(path,parse_dates=['date']).sort_values('date').set_index('date')['close']
px=pd.concat({s:rd('../persistent/stock_data/'+s+'.csv') for s in U},axis=1).sort_index()
vix=rd('../persistent/index_data/VIX.csv').reindex(px.index).ffill()
px=px.loc[:cut]; vix=vix.loc[:cut]
r=px.pct_change(); vr=vix.pct_change()
# Shock indicator is lagged one day to avoid using today's VIX in today's signal.
shock=(vr>vr.rolling(60,min_periods=30).quantile(.75)).shift(1)
f=(-r.rolling(3).sum()).where(shock.astype(bool)); f=f.sub(f.median(axis=1),axis=0)
rows=[]
for d in f.index:
 for h in [1,5,10]:
  y=px.shift(-h).div(px).sub(1).loc[d]; z=pd.concat([f.loc[d],y],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((d,h,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
out=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=out[out.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2026-07','2027-02-24')]:
  q=x.set_index('date').loc[lo:hi].ic; print(lo,len(q),round(q.mean(),6) if len(q) else np.nan,round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
s=f.stack().rename('signal').reset_index(); s.columns=['date','symbol','signal']; s.to_csv('../persistent/factor_signals_miner_1_20270225_vix_shock_reversal.csv',index=False)
w=f.rank(axis=1,pct=True); print('turnover',round(w.diff().abs().mean(axis=1).mean(),6),'signal_dates',int(f.notna().any(axis=1).sum()))
