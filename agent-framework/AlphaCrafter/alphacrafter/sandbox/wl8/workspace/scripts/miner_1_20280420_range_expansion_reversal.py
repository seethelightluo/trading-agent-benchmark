import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-04-19')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].set_index('date').sort_index(); D[s]=x
px=pd.DataFrame({s:x.close for s,x in D.items()}).sort_index(); hi=pd.DataFrame({s:x.high for s,x in D.items()}).reindex(px.index); lo=pd.DataFrame({s:x.low for s,x in D.items()}).reindex(px.index)
r=px.pct_change()
# Expansion-weighted short-term reversal: fade 5d moves more strongly when recent true range expands.
tr=(hi-lo)/px
exp=tr.rolling(5,min_periods=4).mean().div(tr.rolling(20,min_periods=15).mean())
f=(-r.rolling(5,min_periods=5).sum()*exp).shift(1)
f.to_csv('scripts/miner_1_20280420_range_expansion_reversal_signal.csv',index_label='date')
for h in [1,3,5,10]:
 y=px.shift(-h)/px-1; a=[]; ds=[]; ns=[]
 for d in px.index:
  g=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(g)>=8:
   q=spearmanr(g.f,g.y).statistic
   if np.isfinite(q): a.append(q); ds.append(d); ns.append(len(g))
 a=np.array(a); print('h',h,'dates',len(a),'rows',sum(ns),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==1:
  for lab,fn in [('2026',lambda d:d.year==2026),('2027',lambda d:d.year==2027),('2028',lambda d:d.year>=2028),('recent180',lambda d:d>=END-pd.Timedelta(days=180))]:
   z=a[[i for i,d in enumerate(ds) if fn(d)]]; print(lab,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
