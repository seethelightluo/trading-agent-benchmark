import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
syms=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in syms:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<80:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);fs[s]=d.set_index('date').sort_index()
p=pd.DataFrame({s:d.close for s,d in fs.items()})
vix=pd.read_csv('../persistent/index_data/VIX.csv');vix['date']=pd.to_datetime(vix['date']);vix=vix.set_index('date')['close'].reindex(p.index).ffill()
# High-volatility conditional short-term reversal, with one-day lag; cross-sectionally centered VIX multiplier.
zv=((vix-vix.rolling(60,min_periods=30).mean())/(vix.rolling(60,min_periods=30).std()+1e-8)).clip(-2,2)
factor=(-p.pct_change(5)*(1+0.5*zv.clip(lower=0))).shift(1)
rows=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
r=pd.DataFrame(rows,columns=['date','n','ic']).dropna();print('symbols',len(fs),'dates',len(r),'avgN',r.n.mean(),'coverage',len(fs)/15)
for h in [5,10,20]:
 a=[]
 for dt in factor.index:
  q=pd.concat([factor.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(q)>=8:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna();print('horizon',h,'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean().mean())
for label,sub in [('2026+',r[r.date>='2026-01-01']),('2027+',r[r.date>='2027-01-01']),('2028YTD',r[r.date>='2028-01-01'])]:print(label,'dates',len(sub),'IC %.6f ICIR %.6f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1)))
r.to_csv('scripts/miner_1_20280518_vix_conditional_reversal_signal.csv',index=False)
