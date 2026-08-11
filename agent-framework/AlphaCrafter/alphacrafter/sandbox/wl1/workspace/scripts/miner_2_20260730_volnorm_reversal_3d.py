import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 fn='../persistent/index_data/'+s+'.csv';fn=fn if os.path.exists(fn) else '../persistent/stock_data/'+s+'.csv';x=pd.read_csv(fn);x.date=pd.to_datetime(x.date);x=x.sort_values('date').drop_duplicates('date').set_index('date');D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index().loc[:'2026-07-15'];r=p.pct_change();vol=np.sqrt(r.pow(2).rolling(20).sum());f=-r.rolling(3).sum().div(vol.replace(0,np.nan))
for h in [1,3,5,10]:
 ic=[];ns=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('a'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:ic.append(z.a.corr(z.y));ns.append(len(z))
 q=pd.Series(ic);print('H',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
ranks=f.rank(axis=1,pct=True);print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover',round(ranks.iloc[::10].diff().abs().mean(axis=1).mean(),4),'period',p.index.min(),p.index.max(),'n',len(p))
