import numpy as np, pandas as pd
from scipy.stats import spearmanr
# Single idea: high-range close-location impulse. Persistent closes near the daily high/low
# are weighted more when their intraday range is elevated versus the asset's own 60-observation norm.
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2029-06-27'; ROOT='../persistent/stock_data'
D={a:pd.read_csv(f'{ROOT}/{a}.csv').set_index('date').sort_index().loc[:END] for a in A}; ix=sorted(set().union(*[set(d.index) for d in D.values()])); c=pd.DataFrame({a:D[a].reindex(ix).close for a in A}); hi=pd.DataFrame({a:D[a].reindex(ix).high for a in A}); lo=pd.DataFrame({a:D[a].reindex(ix).low for a in A})
rng=(hi-lo).replace(0,np.nan); clv=((2*c-hi-lo)/rng).clip(-1,1); relrange=(rng/c).rolling(5,min_periods=4).mean().div((rng/c).rolling(60,min_periods=30).mean()).clip(0,4); f=clv.rolling(5,min_periods=4).mean()*relrange; vis=c.index[c.index<=END]
print('FACTOR high_range_close_location_impulse_5_60obs cutoff',vis[-1],'assets',len(A),'cells',int(f.loc[vis].notna().sum().sum()),'of',len(vis)*15)
def stat(sub,h):
 fw=c.shift(-h).div(c)-1; vals=[];ns=[];turn=[]; prev=None
 for t in sub:
  z=pd.concat([f.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  q=f.loc[t].rank(); z2=pd.concat([q,prev],axis=1).dropna() if prev is not None else pd.DataFrame()
  if len(z2)>=8: turn.append(1-spearmanr(z2.iloc[:,0],z2.iloc[:,1]).statistic)
  prev=q
 x=np.array(vals); return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns),np.mean(turn)
for h in [1,5,10,20]:
 x=stat(vis,h); print('H',h,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4),'mean_n',round(x[4],2),'coverage',round(f.loc[vis].notna().mean().mean(),4),'turn',round(x[5],4))
for lab,sub in [('2020_21',vis[vis<'2022-01-01']),('2022_23',vis[(vis>='2022-01-01')&(vis<'2024-01-01')]),('2024_25',vis[(vis>='2024-01-01')&(vis<'2026-01-01')]),('2026_current',vis[vis>='2026-01-01'])]:
 x=stat(sub,5); print('REGIME_5D',lab,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4))
