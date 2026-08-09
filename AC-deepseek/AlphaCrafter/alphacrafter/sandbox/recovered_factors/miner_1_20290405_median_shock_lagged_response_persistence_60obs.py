"""Single idea: cross-asset median-shock lagged-response persistence.
After an unusually large benchmark-wide move, assets may display differentiated
next-session responses. The signal is each asset's rolling beta to the prior
session's tail median return; it is deliberately a lagged response rather than
contemporaneous market beta."""
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-04-04')
def ld(a,c='close',idx=False):
 p=('../persistent/index_data/' if idx else '../persistent/stock_data/')+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,c].astype(float)
p={a:ld(a) for a in A};r=pd.DataFrame({a:p[a].pct_change() for a in A});med=r.median(axis=1)
# Tail only: condition lagged benchmark exposures on the 20% largest moves.
sh=med.abs()>med.abs().rolling(60,min_periods=45).quantile(.80); x=med.shift().where(sh.shift())
f=pd.DataFrame({a:r[a].rolling(60,min_periods=15).cov(x)/x.rolling(60,min_periods=15).var() for a in A})
print('CANDIDATE median_shock_lagged_response_persistence_60obs visible_through',END.date(),'assets=15')
best=None
for h in [1,5,10,20]:
 y=pd.DataFrame({a:p[a].shift(-h)/p[a]-1 for a in A});z=[];cv=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:z.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));cv.append(len(q)/15)
 xic=pd.Series(dict(z));ic=xic.mean();ir=ic/xic.std(ddof=1);print(f'H={h} dates={len(xic)} IC={ic:.6f} ICIR={ir:.6f} hit={(xic>0).mean():.4f} coverage={np.mean(cv):.4f} mean_instruments={15*np.mean(cv):.2f}')
 if best is None or abs(ic*ir)>abs(best[1].mean()*(best[1].mean()/best[1].std(ddof=1))):best=(h,xic)
h,xic=best;print('BEST_HORIZON',h)
for n,lo,hi in [('2020-21','2020','2022'),('2022-23','2022','2024'),('2024-25','2024','2026'),('2026-current','2026','2030')]:
 z=xic[(xic.index>=lo)&(xic.index<hi)];print(f'REGIME {n} dates={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
# An unsuccessful candidate is not persisted, so full library-independence
# computation is only needed if the binding IC/ICIR gates are passed.
"""
