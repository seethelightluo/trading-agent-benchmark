import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300: d=get_index_daily_data(s,days=6000)
 return d[['date','close']].copy() if d is not None else None
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
px=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill()
r=np.log(px).diff(); mkt=r.mean(axis=1); res=r.sub(mkt,axis=0)
# Conditional short-horizon reversal: recent residual losses mean-revert most when market dispersion is high,
# while suppressing the signal in extreme VIX panic where trend continuation dominates.
disp=res.rolling(20).std().mean(axis=1)
disp_z=(disp-disp.rolling(252).mean())/(disp.rolling(252).std()+1e-12)
try:
 v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').iloc[:,0].astype(float).reindex(px.index).ffill()
except Exception: v=pd.Series(index=px.index,dtype=float)
vz=(v-v.rolling(252).mean())/(v.rolling(252).std()+1e-12)
# lag all inputs; positive factor predicts forward return
rev=-res.rolling(3).sum()
state=(1+0.5*disp_z.clip(-1,2)).clip(0.25,2.0)*(1-0.35*vz.clip(lower=0,upper=2))
f=rev.mul(state,axis=0).shift(1)
fr=px.shift(-10)/px-1
rows=[]
for dt in px.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): rows.append((dt,ic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mean=q.ic.mean(); sd=q.ic.std(ddof=1); icir=mean/sd*np.sqrt(252/10)
turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('dates',len(q),'avg_n',q.n.mean(),'coverage',len(q)/len(px.index),'IC10',mean,'ICIR10',icir,'hit',(q.ic>0).mean(),'turn',turn)
for n in [120,252,756,1260]:
 z=q.tail(n); print('recent',n,z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252/10))
for h in [5,10,20]:
 yy=px.shift(-h)/px-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(252/h))
f.to_csv('scripts/miner_1_20340331_dispersion_vix_reversal_signal.csv',index_label='date')
q.to_csv('scripts/miner_1_20340331_dispersion_vix_reversal_ic.csv')
