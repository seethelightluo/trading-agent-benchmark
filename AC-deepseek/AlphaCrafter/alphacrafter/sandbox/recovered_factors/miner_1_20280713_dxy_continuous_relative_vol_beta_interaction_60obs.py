"""Continuous DXY beta-volatility interaction; avoids sparse conditional state split."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-07-12')
def load(a,idx=False):
 return pd.read_csv(('../persistent/index_data/' if idx else '../persistent/stock_data/')+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float)
p=pd.DataFrame({a:load(a) for a in A}); r=p.pct_change();d=load('DXY',True).pct_change(); vol=r.rolling(20,min_periods=15).std()
# beta is transmission to DXY; multiply by continuous own relative-vol state, centered per asset.
def beta(x,y,n=60): return x.rolling(n,min_periods=n).cov(y)/y.rolling(n,min_periods=n).var()
b=pd.DataFrame({a:beta(r[a],d) for a in A})
# Current 20d vol relative to its trailing 60d mean, a continuous state rather than binary split.
state=vol/vol.rolling(60,min_periods=45).mean()-1
f=b*state
print('CANDIDATE dxy_continuous_relative_vol_beta_interaction_60obs visible_through',END.date(),'assets',len(A))
best=None
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ics=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8: ics.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));ns.append(len(q))
 x=pd.Series(dict(ics)); ic=x.mean();ir=ic/x.std(ddof=1)
 print(f'H={h} dates={len(x)} IC={ic:.6f} ICIR={ir:.6f} hit={(x>0).mean():.4f} coverage={np.mean(ns)/15:.4f} mean_instruments={np.mean(ns):.2f}')
 if best is None or abs(ic*ir)>abs(best[1].mean()*(best[1].mean()/best[1].std(ddof=1))):best=(h,x)
h,x=best;print('BEST_HORIZON',h)
for n,lo,hi in [('2020','2020','2021'),('2021-22','2021','2023'),('2023-24','2023','2025'),('2025-current','2025','2030')]:
 z=x[(x.index>=lo)&(x.index<hi)]; print(f'REGIME {n} dates={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(tos):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
# Correlation to closest directly related active DXY factor, reconstructed identically.
med=vol.rolling(60,min_periods=45).median(); old=pd.DataFrame(index=r.index,columns=A,dtype=float)
for a in A:
 z=pd.concat([r[a].rename('x'),d.rename('d'),(vol[a]>med[a]).rename('s')],axis=1).dropna(); o=[]
 for i in range(len(z)):
  q=z.iloc[max(0,i-59):i+1]; hi=q[q.s];lo=q[~q.s]
  def bb(v):return v.x.cov(v.d)/v.d.var() if len(v)>=12 and v.d.var()>0 else np.nan
  o.append(bb(hi)-bb(lo))
 old[a]=pd.Series(o,index=z.index).reindex(r.index)
q=pd.concat([f.stack(),old.stack()],axis=1).dropna();print(f'RELATED dxy_relative_vol_regime_beta_spread rho={q.iloc[:,0].corr(q.iloc[:,1],method="spearman"):.6f} cells={len(q)}')
