import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(p.index).ffill()
z=(v-v.rolling(60,min_periods=30).mean())/v.rolling(60,min_periods=30).std(); stress=(1+z.clip(-2,2)/4).clip(.5,1.5)
# short trend-following signal, amplified continuously by volatility stress
f=r.rolling(5,min_periods=5).sum().mul(stress,axis=0)
fr=p.shift(-10)/p-1; rows=[]
for dt in f.index:
 x=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(x)>=8:
  c=x.iloc[:,0].corr(x.iloc[:,1],method='spearman')
  if np.isfinite(c): rows.append((dt,c,len(x)))
df=pd.DataFrame(rows,columns=['date','ic','n'])
print('dates',len(df),'avg_names',round(df.n.mean(),2),'coverage',round(df.n.mean()/15,4),'range',df.date.min(),df.date.max())
for lab,zs in [('all',df),('2026_27',df[(df.date>='2026-01-01')&(df.date<'2028-01-01')]),('2028',df[df.date>='2028-01-01'])]:
 ic=zs.ic.mean(); print(lab,'dates',len(zs),'IC',round(ic,8),'ICIR',round(ic/zs.ic.std(ddof=1),8),'hit',round((zs.ic>0).mean(),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6)); f.to_csv('scripts/miner_2_20280713_vix_stress_trend_signal.csv')
