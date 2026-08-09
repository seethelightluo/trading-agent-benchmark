import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
p=pd.DataFrame(P).sort_index().loc[:'2026-11-18']; r=p.pct_change(fill_method=None)
# Macro-conditioned short reversal: reverse recent 5d return, with stronger signal when VIX has risen
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].sort_index().loc[:'2026-11-18']
# align macro observation through t; signal uses t close and is evaluated t+1
vix=v.reindex(p.index).ffill(); shock=(vix.pct_change(5).clip(-1,1))
base=-r.rolling(5,min_periods=3).sum()
# nonlinear regime multiplier, preserving reversal direction
F=base*(1+0.8*shock.clip(lower=0)).replace([np.inf,-np.inf],np.nan)
F.to_csv('scripts/miner_1_20261119_vix_conditioned_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 y=p.pct_change(h,fill_method=None).shift(-h); vals=[];ns=[];ds=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
 if h==1:
  d=pd.DatetimeIndex(ds)
  for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
   q=a[(d.year>=int(lo))&(d.year<=int(hi))];print('REG',lo,hi,len(q),round(np.nanmean(q),6),round(np.nanmean(q)/np.nanstd(q,ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'period',p.index.min(),p.index.max(),'assets',p.shape[1])
