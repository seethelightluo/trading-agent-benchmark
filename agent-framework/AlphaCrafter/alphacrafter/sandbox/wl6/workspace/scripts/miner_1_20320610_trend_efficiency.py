import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-06-09')
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.astype(float) for s in U},axis=1).sort_index().loc[:cut]
r=P.pct_change()
# Trend efficiency: directional 20-day displacement divided by total absolute daily movement; rewards persistent trends, penalizes choppy paths.
f=(P/P.shift(20)-1)/(r.abs().rolling(20,min_periods=16).sum()+1e-12)
print('cutoff',P.index.max().date(),'dates',len(P),'assets',P.shape[1])
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; vals=[];ns=[];ds=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt].rename('x'),fr.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=z.x.corr(z.y,method='spearman')
   if np.isfinite(q): vals.append(q);ns.append(len(z));ds.append(dt)
 a=np.array(vals); print('h',h,'valid_dates',len(a),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
 q=pd.Series(a,index=ds); print('regimes',q.groupby(q.index.year).mean().round(6).to_dict())
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
