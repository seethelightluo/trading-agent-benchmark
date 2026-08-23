import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); ret10=r.rolling(10).sum(); ret20=r.rolling(20).sum(); risk20=r.rolling(20).std()*np.sqrt(20)
breadth=ret20.median(axis=1)
# stress/bearish breadth: short-term reversal; constructive breadth: medium trend
f=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
f.loc[breadth<0]=(-ret10/(risk20+1e-6)).loc[breadth<0]
f.loc[breadth>=0]=(ret20/(risk20+1e-6)).loc[breadth>=0]
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(y[ok]),ok.mean()))
d=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for n,z in [('all',d),('2020_24',d[d.date<'2025-01-01']),('2025_26',d[(d.date>='2025-01-01')&(d.date<'2027-01-01')]),('2027_28',d[d.date>='2027-01-01']),('recent',d[d.date>='2027-09-07'])]:
 if len(z): print(n,'dates',len(z),'avg_names',round(z.coverage.mean()*15,2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(d.coverage.mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'range',d.date.min(),d.date.max(),'instruments',len(px),'bear_days',int((breadth<0).sum()))
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20280907_breadth_switch_signal.csv',index=False)
