import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();px[s]=d.set_index('date').close
p=pd.DataFrame(px).sort_index().ffill();r=p.pct_change(); m=r.mean(axis=1)
# Relative reversal: reverse each asset's 10d return after removing common cross-asset movement.
beta=r.rolling(60).cov(m)/(m.rolling(60).var()+1e-8); resid=r.sub(beta.mul(m,axis=0)); f=-resid.rolling(10).sum()/(resid.rolling(20).std()*np.sqrt(20)+1e-8)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],x[ok].corr(y[ok]),ok.mean()))
d=pd.DataFrame(rows,columns=['date','ic','coverage']).dropna()
for n,z in [('all',d),('2020_24',d[d.date<'2025-01-01']),('2025_26',d[(d.date>='2025-01-01')&(d.date<'2027-01-01')]),('2027_28',d[d.date>='2027-01-01'])]: print(n,len(z),round(z.coverage.mean()*15,2),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1),6),round((z.ic>0).mean(),4))
rank=f.rank(axis=1,pct=True);print('coverage',d.coverage.mean(),'turnover',rank.diff().abs().mean().mean(),'range',d.date.min(),d.date.max(),'instruments',len(px))
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20280810_residual_reversal_signal.csv',index=False)
