import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); D[s]=d['close']
p=pd.DataFrame(D); r=p.pct_change(); f=r.rolling(5).sum()-r.shift(5).rolling(15).sum()
for h in [1,5,10]:
 fw=p.shift(-h)/p-1; arr=[]
 for t in f.index:
  z=pd.concat([f.loc[t].rename('f'),fw.loc[t].rename('y')],axis=1).dropna()
  if len(z)>=8:
   c=z.f.corr(z.y,method='spearman')
   if pd.notna(c): arr.append((t,c,len(z)))
 q=pd.DataFrame(arr,columns=['date','ic','n']).set_index('date'); print('h',h,'dates',len(q),'avg_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
 for lo,hi in [('2020-01-01','2023-01-01'),('2023-01-01','2025-01-01'),('2025-01-01','2028-01-01')]:
  a=q[(q.index>=lo)&(q.index<hi)].ic; print(lo,len(a),a.mean(),a.mean()/a.std(ddof=1) if len(a)>2 else np.nan)
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
