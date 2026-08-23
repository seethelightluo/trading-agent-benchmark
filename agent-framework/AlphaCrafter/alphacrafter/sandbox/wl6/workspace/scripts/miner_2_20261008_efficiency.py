import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-10-07')
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d['date']=pd.to_datetime(d.date); d=d[d.date<=cutoff].sort_values('date').set_index('date'); px[a]=d.close.astype(float)
px=pd.DataFrame(px).sort_index(); rets=px.pct_change(); net=px.pct_change(10); path=rets.abs().rolling(10,min_periods=8).sum(); f=net/path; fwd=px.shift(-1)/px-1
rows=[]
for date in f.index:
 z=pd.concat([f.loc[date],fwd.loc[date]],axis=1).dropna()
 if len(z)>=8: rows.append((date,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(r),'avg_names',round(r.n.mean(),3),'coverage',round(r.n.mean()/15,4),'period',r.index.min().date(),r.index.max().date())
print('daily IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for h in [5,10,20]:
 fw=px.shift(-h)/px-1; vals=[]
 for date in f.index:
  z=pd.concat([f.loc[date],fw.loc[date]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 vals=np.array(vals); print('%dd IC %.8f ICIR %.8f n %d'%(h,vals.mean(),vals.mean()/vals.std(),len(vals)))
for label,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-10-07')]:
 q=r.loc[a:b].ic; print(label,len(q),'IC',q.mean(),'ICIR',q.mean()/q.std())
for name,g in [('momentum',px.pct_change(20)),('reversal',-px.pct_change(5))]: print('rho',name,f.rank(axis=1).corrwith(g.rank(axis=1),axis=1).mean())
