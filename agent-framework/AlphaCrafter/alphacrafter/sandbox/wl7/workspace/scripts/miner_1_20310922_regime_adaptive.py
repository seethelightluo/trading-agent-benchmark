import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); ew=r.mean(axis=1)
ret20=P.pct_change(20); vol=r.rolling(30,min_periods=20).std()
# Regime-adaptive relative signal: mean-revert residual momentum in weak markets,
# follow residual momentum in non-weak markets. Regime uses only trailing data.
res=ret20-ret20.mean(axis=1).values[:,None]
base=res.div(vol.replace(0,np.nan))
market60=ew.rolling(60,min_periods=45).sum()
q=market60.rolling(252,min_periods=126).quantile(.333)
weak=(market60<q)
sig=base.where(weak, -base).shift(1)
# cross-sectional standardization is not needed for rank IC, but keeps scale interpretable
def ev(h):
 y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&y.loc[dt].notna()
  if v.sum()>=8:
   vals.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman')); rows.append((dt,vals[-1],int(v.sum())))
 a=pd.Series(vals); return a,rows
for h in [1,5,10,20]:
 a,rows=ev(h); print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rows]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rows=ev(10)
print('history_dates',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%((sig.notna()).mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
# thirds by chronological regime for robustness
n=len(a); print('third_IC',[round(a.iloc[i:j].mean(),8) for i,j in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20310922_regime_adaptive_ic_10d.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310922_regime_adaptive_signal.csv',index=False)
