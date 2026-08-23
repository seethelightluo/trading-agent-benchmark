import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# VIX is observation-only macro data, loaded from index_data rather than tradable stock data.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(p.index).ffill()
vz=(v-v.rolling(60,min_periods=30).mean())/v.rolling(60,min_periods=30).std()
stress=(1+vz.clip(-2,2)/4).clip(.5,1.5)
raw=-r.rolling(5,min_periods=5).sum(); f=raw.mul(stress,axis=0)
rows=[]
for dt in f.index:
 x=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(x)>=8:
  c=x.iloc[:,0].corr(x.iloc[:,1],method='spearman')
  if np.isfinite(c): rows.append((dt,c,len(x)))
df=pd.DataFrame(rows,columns=['date','ic','n'])
print('dates',len(df),'avg_names',round(df.n.mean(),2),'coverage',round(df.n.mean()/15,4),'range',df.date.min(),df.date.max())
for lab,z in [('all',df),('2020_24',df[df.date<'2025-01-01']),('2025_26',df[(df.date>='2025-01-01')&(df.date<'2027-01-01')]),('2027_28',df[df.date>='2027-01-01'])]:
 ic=z.ic.mean(); ir=ic/z.ic.std(ddof=1)
 print(lab,'dates',len(z),'IC',round(ic,8),'ICIR',round(ir,8),'hit',round((z.ic>0).mean(),4))
rank=f.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean().mean(),6))
f.to_csv('scripts/miner_2_20280601_vix_conditioned_reversal_ic.csv',index=False)
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_2_20280601_vix_conditioned_reversal_signal.csv',index=False)
