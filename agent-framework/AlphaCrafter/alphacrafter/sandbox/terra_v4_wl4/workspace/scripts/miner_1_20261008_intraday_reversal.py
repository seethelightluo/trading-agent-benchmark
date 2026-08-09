import numpy as np, pandas as pd
from pathlib import Path
root=Path('../persistent'); end=pd.Timestamp('2026-10-07')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv(root/'stock_data'/(s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date').loc['2020-01-01':end]
 return d
D={s:load(s) for s in syms}; px=pd.concat({s:d.close for s,d in D.items()},axis=1,sort=True); op=pd.concat({s:d.open for s,d in D.items()},axis=1,sort=True)
f=-(px/op-1).replace([np.inf,-np.inf],np.nan)
def ev(h):
 y=px.shift(-h).div(px)-1; z=f.where(f.notna()&y.notna()); yy=y.where(z.notna()); n=z.notna().sum(axis=1); xm=z.sum(axis=1).div(n); ym=yy.sum(axis=1).div(n); cov=((z-xm.values[:,None])*(yy-ym.values[:,None])).sum(axis=1); sx=((z-xm.values[:,None])**2).sum(axis=1).pow(.5); sy=((yy-ym.values[:,None])**2).sum(axis=1).pow(.5); ic=cov.div(sx*sy); return pd.DataFrame({'ic':ic[n>=8],'n':n[n>=8]})
q=ev(1); print('factor=intraday reversal; dates',len(q),'mean names',q.n.mean(),'coverage',q.n.mean()/15); print('IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=q.loc[a:b]; print(a+'-'+b,'dates',len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
for h in [5,10]:
 z=ev(h); print('horizon',h,'dates',len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std()))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'validation_end',q.index.max().date()); f.to_csv('scripts/miner_1_20261008_intraday_reversal_signal.csv',index_label='date')
