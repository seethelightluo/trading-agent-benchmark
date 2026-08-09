import numpy as np, pandas as pd
from pathlib import Path
root=Path('../persistent'); end=pd.Timestamp('2026-10-07')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv(root/'stock_data'/(s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date').loc[:end]
 return d
D={s:load(s) for s in syms}
px=pd.concat({s:d.close for s,d in D.items()},axis=1); op=pd.concat({s:d.open for s,d in D.items()},axis=1)
# Intraday reversal: negative same-session open-to-close return, known after close t, predicts t+1 close return.
f=-(px/op-1).replace([np.inf,-np.inf],np.nan); y=px.shift(-1).div(px)-1

def ev(h):
 yy=px.shift(-h).div(px)-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
q=ev(1)
print('factor=intraday reversal; dates',len(q),'mean names',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
print('IC %.6f ICIR %.6f hit %.4f'% (q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=q.loc[a:b]; print(a+'-'+b,'dates',len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
for h in [5,10]:
 z=ev(h); print('horizon',h,'dates',len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std()))
ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean(),'validation_end',q.index.max().date())
f.to_csv('scripts/miner_1_20261008_intraday_reversal_signal.csv',index_label='date')
