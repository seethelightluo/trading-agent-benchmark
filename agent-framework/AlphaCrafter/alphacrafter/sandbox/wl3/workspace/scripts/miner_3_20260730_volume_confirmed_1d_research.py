import pandas as pd, numpy as np
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
P={};V={}
for s in symbols:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date').set_index('date'); P[s]=d.close; V[s]=d.volume
px=pd.DataFrame(P); vol=pd.DataFrame(V); r=px.pct_change(); rv=r.rolling(20,min_periods=15).std(); vs=vol.div(vol.rolling(20,min_periods=15).median()).replace([np.inf,-np.inf],np.nan)
sig=(-r/rv).mul(np.sqrt(vs.clip(.5,3))); fwd=r.shift(-1); rows=[]; turnover=[]; prev=None
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z))); ranks=sig.loc[dt].rank(pct=True)
  if prev is not None: turnover.append((ranks-prev).abs().mean())
  prev=ranks
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def rep(a): return len(a),a.ic.mean(),a.ic.std(ddof=1),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean(),a.n.mean()
print('dates IC ICIR hit meanN',rep(q))
for n,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-07-15')]: print(n,rep(q.loc[a:b]))
print('coverage',sig.notna().sum(axis=1).mean()/15,'turnover',np.mean(turnover))
for h in [5,10]:
 yy=px.pct_change(h).shift(-h); zics=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:zics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=pd.Series(zics);print('decay',h,len(a),a.mean(),a.mean()/a.std(ddof=1))
q.to_csv('scripts/miner_3_20260730_volume_confirmed_1d_research.csv')
