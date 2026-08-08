import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-12-16')
def get(s,index=False):
 p='../persistent/index_data/'+s+'.csv' if index else '../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
raw={a:get(a) for a in A}; C=pd.concat([raw[a].close.rename(a) for a in A],axis=1); R=C.pct_change(fill_method=None)
dxy=get('DXY',True).close.reindex(C.index).ffill(); dr=dxy.pct_change(fill_method=None)
# Single idea: higher values identify assets whose returns are resilient on DXY-up days.
up=dr.where(dr>0,0.0); F=-R.rolling(40,min_periods=28).cov(up).div(up.rolling(40,min_periods=28).var(),axis=0)
def quiet(a): return (C[a].pct_change(20,fill_method=None).abs()/R[a].abs().rolling(20,min_periods=15).sum())*(1-R[a].rolling(20,min_periods=15).std().rolling(60,min_periods=40).rank(pct=True))
vix=get('VIX',True).close.reindex(C.index).ffill()
base=C.pct_change(20,fill_method=None)/R.rolling(20,min_periods=15).std()
LIB={'ravmom_20obs':base,'volnorm_reversal_5obs':-C.pct_change(5,fill_method=None)/R.rolling(5,min_periods=4).std(),'volscaled_reversal_1obs':-R/R.rolling(20,min_periods=15).std(),'relative_volume_participation_20d':pd.concat([np.log(raw[a].volume/raw[a].volume.rolling(20,min_periods=15).mean()).rename(a) for a in A],axis=1),'quiet_trend_path_efficiency_20_60':pd.concat([quiet(a).rename(a) for a in A],axis=1),'vix_regime_conditioned_risk_adjusted_trend_20':base.mul(np.where(vix.pct_change(20,fill_method=None)>0,-1,1),axis=0)}
def stats(h, dates=None):
 out=[]; nn=[]; fw=C.shift(-h).div(C).sub(1)
 for d in (F.index if dates is None else dates):
  z=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.r).statistic
   if np.isfinite(q):out.append(q);nn.append(len(z))
 return np.array(out),nn
print('FACTOR=dxy_upside_shock_resilience_beta40; validation_date=2026-12-17; cutoff=2026-12-16')
print('definition: negative 40-observation rolling beta of asset daily returns to positive-only DXY daily returns; higher = resilient to dollar-strength shocks')
print('cells=%d/%d coverage=%.6f'%(F.notna().sum().sum(),F.size,F.notna().mean().mean()))
for h in [1,5,10,20]:
 x,n=stats(h); ir=x.mean()/x.std(ddof=1); print('H=%d dates=%d meanIC=%.6f absIC=%.6f ICIR=%.6f absICIR=%.6f hit=%.4f names=%.2f'%(h,len(x),x.mean(),abs(x.mean()),ir,abs(ir),(x>0).mean(),np.mean(n)))
print('rank_change_turnover=%.6f'%F.rank(axis=1,pct=True).diff().abs().stack().mean())
for label,lo,hi in [('2020','2020-01-01','2020-12-31'),('2021-22','2021-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-16')]:
 x,n=stats(10,F.loc[lo:hi].index); print('REGIME=%s dates=%d IC=%.6f ICIR=%.6f hit=%.4f'%(label,len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
mx=0.0
for name,L in LIB.items():
 z=pd.concat([F.stack().rename('f'),L.stack().rename('l')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); q=spearmanr(z.f,z.l).statistic; mx=max(mx,abs(q)); print('LIBRARY %s rho=%.10f cells=%d'%(name,q,len(z)))
print('max_abs_library_correlation=%.10f'%mx)
