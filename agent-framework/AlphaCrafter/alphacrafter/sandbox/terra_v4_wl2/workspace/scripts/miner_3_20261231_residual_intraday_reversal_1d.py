import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-31')
O={}; C={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]; O[s]=d.open; C[s]=d.close
op=pd.concat(O,axis=1,sort=False).reindex(columns=U); cl=pd.concat(C,axis=1,sort=False).reindex(columns=U)
intr=cl/op-1; f=-(intr-intr.mean(axis=1).values[:,None]); f=pd.DataFrame(f,index=cl.index,columns=U); y=cl.pct_change().shift(-1)
a=[]; dates=[]; ns=[]
for i in range(len(cl)-1):
 q=pd.concat([f.iloc[i].rename('signal'),y.iloc[i].rename('forward')],axis=1).dropna()
 if len(q)>=8 and q.signal.nunique()>1:
  r=spearmanr(q.signal,q.forward).statistic
  if np.isfinite(r): a.append(r); dates.append(cl.index[i]); ns.append(len(q))
a=np.array(a); dt=pd.DatetimeIndex(dates)
def stat(z): return float(np.mean(z)),float(np.mean(z)/np.std(z,ddof=1))
print('candidate residual_intraday_reversal_1d dates',len(a),'avg_n',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(stat(a)[0],stat(a)[1],np.mean(a>0)))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 z=a[(dt>=pd.Timestamp(lo))&(dt<=pd.Timestamp(hi))]; print('regime',lo,'n',len(z),'IC %.6f ICIR %.6f'%stat(z))
valid=f.notna()&y.notna(); turn=float(np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)))
print('factor_coverage',round(float(f.notna().sum().sum()/f.size),5),'mean_date_coverage',round(float(valid.sum(axis=1).div(len(U)).mean()),5),'turnover',round(turn,5))
out=Path('scripts/miner_3_20261231_residual_intraday_reversal_1d_signal.csv'); q=f.stack(future_stack=False,dropna=False).rename('signal').reset_index(); q.columns=['date','symbol','signal']; q.to_csv(out,index=False); print('artifact',out,'rows',len(q))
