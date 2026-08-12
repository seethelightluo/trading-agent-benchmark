import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-03-07')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
L=30; trend=r.rolling(L,min_periods=20).sum(); up=r.clip(lower=0).rolling(L,min_periods=20).std(); dn=(-r.clip(upper=0)).rolling(L,min_periods=20).std(); f=(trend*(up+.002)/(dn+.002)).shift(1).clip(-20,20)
rows=[]
for h in [5,10,20]:
 I=[]; Ns=[]; ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: I.append(spearmanr(q.f,q.y).statistic);Ns.append(len(q));ds.append(px.index[i])
 a=np.array(I); ds=pd.DatetimeIndex(ds); print('h',h,'dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,mask in [('2020-25',ds<pd.Timestamp('2026-01-01')),('2026+',ds>=pd.Timestamp('2026-01-01')),('2027+',ds>=pd.Timestamp('2027-01-01')),('2028+',ds>=pd.Timestamp('2028-01-01'))]:
  z=a[mask]; print(' ',lab,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift()).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
for d in f.index:
 for s in U: rows.append({'date':d,'symbol':s,'signal':f.loc[d,s]})
pd.DataFrame(rows).to_csv('scripts/miner_1_20290308_updown_vol_balance_signal.csv',index=False)
