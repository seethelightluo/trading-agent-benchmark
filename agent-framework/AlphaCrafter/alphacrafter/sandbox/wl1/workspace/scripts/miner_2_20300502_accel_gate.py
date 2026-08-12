import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  d=d[['date','close']].copy();d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close
p=pd.DataFrame(px).sort_index().ffill(); ret=p.pct_change()
sig=(p.pct_change(20)-p.pct_change(60)/3)/(ret.rolling(40).std()*np.sqrt(20))
sig=sig.where(p.pct_change(120)>-0.05,sig*.25).shift(1)
def calc(h, q=sig):
 fr=p.pct_change(h).shift(-h);a=[];ns=[]
 for dt in q.index:
  x=pd.concat([q.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(x)>=8:a.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'));ns.append(len(x))
 a=np.array(a);return len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)
print('rows',len(p),'assets',p.shape[1],'range',p.index.min(),p.index.max())
for h in [1,5,10,20]:print(h,calc(h))
r=sig.rank(axis=1,pct=True);print('coverage',sig.notna().mean().mean(),'turnover',r.diff().abs().mean(axis=1).dropna().mean(),'signal_rows',int(sig.notna().sum().sum()))
for label,start in [('2027+','2027-01-01'),('2029+','2029-01-01'),('2030','2030-01-01')]:
 q=sig.loc[start:];print(label,calc(1,q))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20300502_accel_gate_signal.csv',index=False)
