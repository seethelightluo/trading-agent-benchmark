import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
# Risk-adjusted medium-horizon momentum: 60d return scaled by 20d volatility.
mom=p.pct_change(60); vol=r.rolling(20).std(); f=(mom/vol).rank(axis=1,pct=True)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(y[ok]),ok.mean()))
d=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for n,z in [('all',d),('2020_24',d[d.date<'2025-01-01']),('2025_26',d[(d.date>='2025-01-01')&(d.date<'2027-01-01')]),('2027_28',d[d.date>='2027-01-01'])]:
 print(n,len(z),round(z.coverage.mean()*15,2),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1),6),round((z.ic>0).mean(),4))
print('turnover',f.diff().abs().mean().mean())
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20280601_riskadj_momentum_signal.csv',index=False)
print('dates',d.date.min(),d.date.max(),'instruments',len(px))
