import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
ret=r.rolling(20).sum(); dn=r.where(r<0,0).rolling(20).std(); f=ret/(dn+1e-6)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(y[ok]),ok.mean()))
d=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for n,z in [('all',d),('2020_24',d[d.date<'2025-01-01']),('2025_26',d[(d.date>='2025-01-01')&(d.date<'2027-01-01')]),('2027_28',d[d.date>='2027-01-01'])]:
 print(n,'dates',len(z),'avg_names',round(z.coverage.mean()*15,2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(d.coverage.mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'range',d.date.min(),d.date.max(),'instruments',len(px))
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20280629_downside_adjusted_momentum_signal.csv',index=False)
