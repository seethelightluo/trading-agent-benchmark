import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-16'); P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float); P[s]=d[d.index<=cut]
P=pd.DataFrame(P).sort_index(); ret=P.pct_change();
for w in [5,10,20,60]:
 raw=P.pct_change(w); F=raw.sub(raw.median(axis=1),axis=0)
 print('WINDOW',w)
 for h in [1,5,10]:
  Y=P.shift(-h).div(P)-1; vals=[]; ns=[]
  for dt in P.index:
   z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
   if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman')); ns.append(len(z))
  ic=pd.Series(vals).dropna(); print('H',h,'dates',len(ic),'avgN',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 print('coverage',round(F.notna().sum().sum()/F.size,4))
