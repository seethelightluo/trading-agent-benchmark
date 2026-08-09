import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-10-07'); rows=[]; sig=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]
 r=x.close.pct_change(); eff=r.rolling(20,min_periods=15).sum()/r.abs().rolling(20,min_periods=15).sum()
 # efficiency captures persistent trend; predict next return
 y=x.close.shift(-1)/x.close-1
 q=pd.DataFrame({'f':eff,'y':y}).dropna()
 for d,z in q.iterrows(): rows.append((d,s,z.f,z.y))
 for d,v in eff.dropna().items(): sig.append({'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)})
a=pd.DataFrame(rows,columns=['date','symbol','f','y']); out=[]
for d,g in a.groupby('date'):
 if len(g)>=8:
  c=g.f.corr(g.y,method='spearman')
  if pd.notna(c):out.append((d,c,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']); print('cutoff',C.date(),'dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(len(a)/(len(z)*15),4)); print('IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
for p in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=z.set_index('date').loc[p[0]:p[1]];print('regime',p,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
for h in [5,10]:
 oo=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C];r=x.close.pct_change();f=r.rolling(20,min_periods=15).sum()/r.abs().rolling(20,min_periods=15).sum();y=x.close.shift(-h)/x.close-1
  for d,v in pd.DataFrame({'f':f,'y':y}).dropna().iterrows():oo.append((d,s,v.f,v.y))
 b=pd.DataFrame(oo,columns=['date','s','f','y']);vv=[]
 for d,g in b.groupby('date'):
  if len(g)>=8:
   c=g.f.corr(g.y,method='spearman');
   if pd.notna(c):vv.append(c)
 vv=np.array(vv);print('horizon',h,'dates',len(vv),'IC %.6f ICIR %.6f'%(vv.mean(),vv.mean()/vv.std(ddof=1)))
print('turnover_proxy',a.sort_values(['symbol','date']).groupby('symbol').f.diff().abs().mean())
pd.DataFrame(sig).to_csv('scripts/miner_3_20261008_efficiency_signal.csv',index=False);print('signal_rows',len(sig))
