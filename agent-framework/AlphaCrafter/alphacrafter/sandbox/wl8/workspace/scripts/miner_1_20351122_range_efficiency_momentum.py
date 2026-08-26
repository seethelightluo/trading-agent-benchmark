import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s,days=4200)
    except Exception:
        try: x=get_index_daily_data(s,days=4200)
        except Exception: x=None
    if x is not None and len(x)>300:
        x=x.sort_values('date').drop_duplicates('date').set_index('date'); D[s]=x['close'].astype(float)
print('loaded',len(D),sorted(D))
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); net=p.pct_change(20); path=r.abs().rolling(20).sum(); vol=r.rolling(20).std()*np.sqrt(20); f=net/path/(vol+1e-12); fr=p.shift(-10)/p-1
rows=[]; sigrows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt].rename('f'),fr.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.f.corr(z.y,method='spearman'),len(z)))
  for sym,val in f.loc[dt].items():
   if pd.notna(val): sigrows.append((dt,sym,val))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman'))
 print('decay',h,'ic',np.nanmean(vals),'n',len(vals))
print('dates',len(ic),'avg_n',ic.n.mean(),'coverage',f.notna().sum().sum()/f.size,'meanIC',ic.ic.mean(),'std',ic.ic.std(),'ICIR',ic.ic.mean()/ic.ic.std(),'hit',(ic.ic>0).mean())
for n in [365,750,1260]:
 q=ic.tail(n).ic; print('recent',n,q.mean(),q.mean()/q.std(),len(q))
ranks=f.rank(axis=1,pct=True); to=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8: to.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('turnover',np.mean(to))
pd.DataFrame(sigrows,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20351122_range_efficiency_momentum_signal.csv',index=False); ic.reset_index().to_csv('scripts/miner_1_20351122_range_efficiency_momentum_ic.csv',index=False)
