import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-07-03'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change()
r3=P.pct_change(3); vol=R.rolling(20).std(); disp=R.rolling(5).std().mean(axis=1)
z=(disp-disp.rolling(60).median())/(disp.rolling(60).std()+1e-8)
f=-(r3/(vol+1e-8)).mul((1+0.5*np.clip(z,-2,2)).to_numpy(),axis=0); y=P.shift(-10)/P-1
def calc(yy):
  ics=[]; ns=[]; ds=[]
  for dt in f.index:
    q=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
    if len(q)>=8 and q.iloc[:,0].nunique()>1:
      ics.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q));ds.append(dt)
  return np.asarray(ics),np.asarray(ns),pd.to_datetime(ds)
a,ns,ds=calc(y); print('ALL',len(a),np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-07-03')]:
 q=(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi));b=a[q];print('REG',lo,hi,len(b),np.mean(ns[q]),b.mean(),b.mean()/b.std(ddof=1),np.mean(b>0))
rk=f.rank(axis=1,pct=True); print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',(rk-rk.shift()).abs().mean(axis=1).dropna().mean())
for h in [1,3,5,10,20]:
 aa,_,_=calc(P.shift(-h)/P-1);print('DECAY',h,aa.mean(),aa.mean()/aa.std(ddof=1))
f.to_csv('scripts/miner_1_20280704_dispersion_conditioned_reversal_signal.csv')
print('period',P.index.min().date(),P.index.max().date())
