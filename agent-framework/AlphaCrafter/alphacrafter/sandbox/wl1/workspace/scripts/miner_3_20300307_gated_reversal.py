import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof='2030-03-06'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in syms:
 d=get_stock_daily_data(s,days=2800)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=2800)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d[d.date<=pd.Timestamp(asof)].sort_values('date');px[s]=d.set_index('date').close.astype(float)
close=pd.DataFrame(px).sort_index(); r=close.pct_change(); vol=r.rolling(20,min_periods=15).std(); trend=close.pct_change(60)
# Short-horizon reversal, selectively active only when the 60d trend is not strongly negative.
raw=-close.pct_change(5)/(vol*np.sqrt(5)+.005)
gate=np.clip((trend+0.10)/0.20,0,1)
factor=raw*gate; factor=factor.replace([np.inf,-np.inf],np.nan)
for h in [1,5,10,20]:
 fwd=close.shift(-h)/close-1; rows=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
 for lab,a,b in [('2020-2025','2020','2025-12-31'),('2026-2028','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-03-06')]:
  x=q[(q.index>=a)&(q.index<=b)]
  if len(x):print(f' {lab}: dates={len(x)} IC={x.ic.mean():.6f} ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f}')
out=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20300307_gated_reversal_signal.csv',index=False);print('coverage=',out.symbol.nunique(),'rows=',len(out),'turnover=',factor.rank(pct=True).diff().abs().mean().mean())
