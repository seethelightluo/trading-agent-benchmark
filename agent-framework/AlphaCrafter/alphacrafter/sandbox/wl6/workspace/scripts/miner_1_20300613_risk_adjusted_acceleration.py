import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(P,s+'.csv'))
 d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change()
# Acceleration: intermediate momentum minus long momentum, risk scaled; higher means improving trend
r20=prices/prices.shift(20)-1; r60=prices/prices.shift(60)-1
vol20=ret.rolling(20).std()*np.sqrt(252)
f=(r20-r60)/vol20.replace(0,np.nan)
rows=[]
for i in range(len(prices)-10):
 dt=prices.index[i]; f0=f.iloc[i]; fr=prices.iloc[i+10]/prices.iloc[i]-1
 ok=f0.notna()&fr.notna()
 if ok.sum()>=8: rows.append([dt,spearmanr(f0[ok],fr[ok]).statistic,ok.sum()])
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# averages and date regimes
print('candidate risk_adjusted_acceleration; dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/15)
print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean(),'std',x.ic.std(ddof=1))
print('year',x.assign(y=x.index.year).groupby('y').ic.agg(['mean','count']))
print('decay')
for h in [5,10,20]:
 z=[]
 for i in range(len(prices)-h):
  a=f.iloc[i]; b=prices.iloc[i+h]/prices.iloc[i]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(spearmanr(a[ok],b[ok]).statistic)
 print(h,np.nanmean(z),len(z))
# turnover of cross-sectional ranks, sampled daily
r=f.rank(axis=1,pct=True); turn=(r.diff().abs().mean(axis=1)).dropna(); print('turnover_proxy',turn.mean())
