import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={};V={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 P[s]=d['close']; V[s]=d['volume'] if 'volume' in d else pd.Series(index=d.index,dtype=float)
P=pd.DataFrame(P).sort_index().loc[:'2026-11-18']; V=pd.DataFrame(V).reindex(P.index)
R=P.pct_change(fill_method=None)
# Volume-confirmed trend: recent 10d return weighted by abnormal log-volume, risk-normalized.
# Volume is used only through completed observations; robust clipping avoids crypto/index scale effects.
vr=np.log1p(V).diff().rolling(20,min_periods=15).mean()
base=np.log1p(V).rolling(20,min_periods=15).mean()
volsur=(np.log1p(V)-base).clip(-3,3)
trend=R.rolling(10,min_periods=8).sum()
vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
F=(trend*(1+0.25*volsur)/vol.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
# make sure no same-day return leakage: signal at t uses close through t; forward starts t+1
for h in [1,5,10]:
 ic=[]; ns=[]; Y=P.pct_change(h,fill_method=None).shift(-h)
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],Y.iloc[i]],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.asarray(ic,float); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(F.notna().sum().sum()/(len(F)*15),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-18')]:
 a=[];Y=P.pct_change().shift(-1)
 for i in range(len(P)-1):
  if lo<=str(P.index[i].date())<=hi:
   z=pd.concat([F.iloc[i],Y.iloc[i]],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('REG',lo,hi,len(a),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),6))
F.to_csv('scripts/miner_3_20261119_volume_confirmed_trend_signal.csv',index_label='date')
