import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-02-23')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); fwd=px.shift(-1)/px-1
# Defensive-relative-strength: lagged 10d risk-adjusted return, with risk-off macro gate
rv=r.rolling(20).std(); base=(r.rolling(10).sum()/rv.replace(0,np.nan)).shift(1)
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v[v.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill()
# In high VIX / rising VIX, invert risky asset ranking toward defensive assets via asset-specific beta proxy
risk=np.array([1,1,1,1,1,1.2,1.2,1.2,0,0.5,1.2,1.5,1.5,-0.3,-0.3]); gate=((v>v.rolling(60).median()) | (v.pct_change(5)>0.08)).astype(float)
sig=base*(1-gate.values[:,None]*risk[None,:]*0.35)
# direct robust defensive tilt: subtract cross-sectional risk exposure under gate
rows=[]; dates=[]; ns=[]
for i,dt in enumerate(px.index):
 g=pd.DataFrame({'s':sig.iloc[i].values, 'f':fwd.loc[dt].values}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): rows.append(q); dates.append(dt); ns.append(len(g))
a=np.array(rows); print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.isfinite(sig).sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lab,m in [('2020-22',px.index.year<=2022),('2023-25',(px.index.year>=2023)&(px.index.year<=2025)),('2026',px.index.year==2026),('2027+',px.index.year>=2027),('recent180',px.index>=END-pd.Timedelta(days=180))]:
 z=a[[j for j,d in enumerate(dates) if m[px.index.get_loc(d)]]]; print(lab,round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,len(z))
ranks=pd.DataFrame(sig,index=px.index).rank(axis=1,pct=True); print('turnover_proxy',round(float(ranks.diff().abs().mean(axis=1).dropna().mean()),6))
