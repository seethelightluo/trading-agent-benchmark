import numpy as np,pandas as pd,json
from alphacrafter.sim.utils import get_stock_daily_data
W=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; a=[]
for s in W:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:a.append(pd.DataFrame({'date':pd.to_datetime(d['date']),'symbol':s,'r':pd.to_numeric(d['pct_change'],errors='coerce').values/100}))
x=pd.concat(a).pivot(index='date',columns='symbol',values='r').sort_index(); lag=x.shift(1)
m=lag.rolling(20,min_periods=15).sum(); v20=lag.rolling(20,min_periods=15).std(); v60=lag.rolling(60,min_periods=45).std(); f=m/(v60*np.sqrt(20)+1e-8)*(v20/(v60+1e-8)); fw=x.shift(-1).rolling(10,min_periods=10).sum().shift(-9)
y=pd.concat([pd.DataFrame({'date':f.index,'symbol':s,'factor':f[s].values,'fw':fw[s].values}) for s in f]).dropna().reset_index(drop=True)
art='scripts/miner_1_20291004_compression_trend_signal.csv';y[['date','symbol','factor']].to_csv(art,index=False)
def c(z):
 q=[];n=[]
 for d,g in z.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1:q.append(g.factor.corr(g.fw,method='spearman'));n.append(len(g))
 q=pd.Series(q).dropna();return len(q),np.mean(n),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
print('overall',c(y),'dates',y.date.nunique(),'assets',y.symbol.nunique(),'coverage',y.factor.notna().mean())
for k,mk in [('2026+',y.date.dt.year>=2026),('2028+',y.date.dt.year>=2028),('2029',y.date.dt.year==2029)]:print(k,c(y[mk]))
z=y.sort_values(['symbol','date']);print('turnover',z.groupby('symbol').factor.apply(lambda q:(q.diff().abs()>0.15).mean()).mean())
