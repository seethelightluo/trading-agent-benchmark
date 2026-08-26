import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>40:F[s]=d[['date','close']].drop_duplicates('date').set_index('date')['close'].rename(s)
p=pd.concat(F.values(),axis=1).sort_index().ffill(); r=np.log(p).diff()
rel=np.log(p/p.shift(5)); rel=rel.sub(rel.median(axis=1),axis=0)
vol=r.rolling(20).std()*np.sqrt(20)
f=-(rel/vol.replace(0,np.nan)).shift(1).clip(-8,8)
q=[]; ns=[]; dates=[]
fr1=np.log(p.shift(-1)/p)
for dt in p.index:
 z=pd.concat([f.loc[dt].rename('f'),fr1.loc[dt].rename('r')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:q.append(z.f.corr(z.r,method='spearman'));ns.append(len(z));dates.append(dt)
q=pd.Series(q);ic=q.mean(); ir=ic/q.std(ddof=1)*np.sqrt(252)
print(f'dates={len(q)} avg_n={np.mean(ns):.2f} IC={ic:.8f} ICIR={ir:.8f} hit={(q>0).mean():.4f}')
for h in [3,5,10,20]:
 a=[]; fr=np.log(p.shift(-h)/p)
 for dt in p.index:
  z=pd.concat([f.loc[dt].rename('f'),fr.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:a.append(z.f.corr(z.r,method='spearman'))
 print('H',h,'IC',np.nanmean(a),'dates',len(a))
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(p),'instruments',len(F),'last',p.index.max().date())
for name,sub in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]:print(name,len(sub),sub.mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20301118_relative_reversal5_vol20_signal.csv',index=False)
q.rename('ic').reset_index().to_csv('scripts/miner_2_20301118_relative_reversal5_vol20_ic.csv',index=False)
