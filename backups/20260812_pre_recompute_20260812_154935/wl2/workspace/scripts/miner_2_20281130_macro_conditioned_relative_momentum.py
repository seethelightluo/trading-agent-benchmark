import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; U=[s for s in U if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<120:d=get_index_daily_data(s,4000)
 if d is not None and len(d)>=120:D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
def macro(n):return pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(p.index).ffill()
v=macro('VIX');d=macro('DXY'); stress=((v>v.rolling(60,min_periods=30).median())|(d.pct_change(20)>0)).astype(float)
raw=r.rolling(20,min_periods=15).sum(); centered=raw.sub(raw.median(axis=1),axis=0);f=(centered.mul(1+.75*stress,axis=0)).shift(1)
print('assets',len(D),'rows',len(p),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=np.log(p).shift(-h)-np.log(p);a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a);print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20281130_macro_conditioned_relative_momentum_signal.csv',index=False)
