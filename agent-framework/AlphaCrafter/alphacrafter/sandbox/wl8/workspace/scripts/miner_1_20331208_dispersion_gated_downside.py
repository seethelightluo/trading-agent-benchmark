import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fn(s,days=5000)
            if d is not None and len(d)>=100:return d
        except Exception: pass
px={s:get(s).set_index('date')['close'] for s in U if get(s) is not None}
P=pd.DataFrame(px).sort_index(); r=np.log(P).diff()
# downside-volatility-normalized 5d reversal, lag-safe dispersion gate
loss=(-r.clip(upper=0)).rolling(30,min_periods=15).std()
base=((-np.log(P/P.shift(5)).clip(upper=0))/(loss.rolling(5,min_periods=3).mean()*np.sqrt(5))).rolling(3,min_periods=2).mean()
disp=r.rolling(20,min_periods=12).std().mean(axis=1)
# only amplify during historically high cross-asset dispersion; threshold uses expanding past only
threshold=disp.shift(1).rolling(252,min_periods=60).quantile(.65)
gate=(disp.shift(1)>threshold).astype(float)
f=base.mul(0.5+gate,axis=0)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(ic):rows.append((dt,len(z),ic))
out=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
sig=f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'});sig.to_csv('scripts/miner_1_20331208_dispersion_gated_downside_signal.csv',index=False);out.to_csv('scripts/miner_1_20331208_dispersion_gated_downside_ic.csv')
for name,ser in [('full',out.ic),('365d',out.ic.tail(365)),('750d',out.ic.tail(750)),('1260d',out.ic.tail(1260))]:print(name,'dates',len(ser),'IC',ser.mean(),'ICIR',ser.mean()/ser.std(ddof=1),'hit',(ser>0).mean())
print('assets',len(P.columns),'avgN',out.n.mean(),'coverage',len(out)*out.n.mean()/(len(P.index)*len(U)),'turnover_proxy',f.rank(axis=1,pct=True).diff().abs().stack().mean())
for h in [1,5,10,20]:
 yy=np.log(P.shift(-h)/P);q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(q))
print('high_gate_share',gate.mean())
