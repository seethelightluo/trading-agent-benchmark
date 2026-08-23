import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std()
# Relative shock: reverse each asset's 5d move relative to contemporaneous universe median,
# normalized by idiosyncratic volatility; all inputs available at t.
rel=r.rolling(5).sum().sub(r.rolling(5).sum().median(axis=1),axis=0)
f=-(rel/(vol*np.sqrt(5)+1e-12)).rank(axis=1,pct=True)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; fr=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&fr.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(fr[ok]),ok.mean()))
d=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for lab,z in [('all',d),('2020_24',d[d.date<'2025-01-01']),('2025_26',d[(d.date>='2025-01-01')&(d.date<'2027-01-01')]),('2027_28',d[d.date>='2027-01-01'])]:
 print(lab,len(z),round(z.coverage.mean()*15,2),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1),6),round((z.ic>0).mean(),4))
print('turnover',round(f.diff().abs().mean().mean(),6)); f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_2_20280420_relative_shock_signal.csv',index=False)
