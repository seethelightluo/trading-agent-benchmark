"""Miner 1: stable liquidity participation factor, one interpretable volume-dispersion idea."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; px={}; vv={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 px[a]=pd.to_numeric(d.close,errors='coerce'); vv[a]=pd.to_numeric(d.get('volume'),errors='coerce').replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vv).reindex(P.index); R=P.pct_change(fill_method=None); med=R.median(axis=1)
# One idea: predictable/steady participation, measured as negative dispersion of log relative volume.
lrv=np.log(V/V.rolling(20,min_periods=15).mean())
F=-lrv.rolling(20,min_periods=15).std()
# Reconstruct all currently admitted signals for mandatory library diversification check.
s20=R.rolling(20,min_periods=15).std(); trend=P.pct_change(20,fill_method=None)/s20; rev5=-P.pct_change(5,fill_method=None)/R.rolling(5,min_periods=4).std(); rev1=-R/s20
rv=lrv; eff=P.pct_change(20,fill_method=None).abs()/R.abs().rolling(20,min_periods=15).sum(); vp=s20.rolling(60,min_periods=40).rank(pct=True)
def ac(x):
 x=x.dropna(); return x.autocorr(1) if len(x)>=15 else np.nan
ia=-R.rolling(20,min_periods=15).apply(ac,raw=False)
neg=med<0; db=pd.DataFrame({a:R[a].where(neg).rolling(40,min_periods=12).cov(med.where(neg))/med.where(neg).rolling(40,min_periods=12).var() for a in A})
trans=ia*np.log(s20.rolling(5,min_periods=4).mean()/s20).clip(-2,2)
idv=-(R.sub(med,axis=0)).rolling(20,min_periods=15).std()
LIB={'risk_adjusted_trend_20d':trend,'ravmom_20obs':trend,'volnorm_reversal_5obs':rev5,'volscaled_reversal_1obs':rev1,'relative_volume_participation_20d':rv,'quiet_trend_path_efficiency_20_60':eff*(1-vp),'inverse_lag1_return_autorrelation_20':ia,'downside_cross_asset_beta_resilience_40':db,'volatility_transition_serial_resilience_20':trans,'inverse_idiosyncratic_volatility_20':idv}
try:
 d=get_index_daily_data('VIX',5000).copy();d.date=pd.to_datetime(d.date);vx=pd.to_numeric(d.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill();vr=vx.pct_change();up=vr>0
 LIB['vix_regime_conditioned_risk_adjusted_trend_20']=trend.where(vx.pct_change(20)<=0,-trend)
 LIB['vix_upside_shock_beta_resilience_40']=pd.DataFrame({a:-(R[a].where(up).rolling(40,min_periods=12).cov(vr.where(up))/vr.where(up).rolling(40,min_periods=12).var()) for a in A})
except Exception as e: print('VIX_ERROR',repr(e))
def stat(h,lo=None,hi=None):
 fw=P.shift(-h)/P-1; out=[]; ns=[]; ix=F.index if lo is None else F.index[(F.index>=lo)&(F.index<=hi)]
 for t in ix:
  z=pd.concat([F.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append(q);ns.append(len(z))
 q=pd.Series(out,dtype=float)
 if len(q)<2:return len(q),np.nan,np.nan,np.nan,np.nan,np.nan
 return len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),np.mean(ns),min(ns)
def f(x):return 'NA' if not np.isfinite(x) else f'{x:.6f}'
print('FACTOR stable_liquidity_participation_20 ENDPOINT',P.index.max().date(),'PERIOD',P.index.min().date(),P.index.max().date(),'ASSETS',len(A))
print('COVERAGE',int(F.notna().sum().sum()),'OF',F.size,'RATE',f(F.notna().mean().mean()))
for h in [1,5,10,20]:
 x=stat(h);print('H',h,'DATES',x[0],'IC',f(x[1]),'ICIR',f(x[2]),'HIT',f(x[3]),'MEAN_NAMES',f(x[4]),'MIN_NAMES',x[5])
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027YTD','2027-01-01','2027-12-31')]:
 x=stat(10,lo,hi);print('REGIME10',n,'DATES',x[0],'IC',f(x[1]),'ICIR',f(x[2]),'HIT',f(x[3]))
print('TURNOVER',f(F.rank(axis=1,pct=True).diff().abs().stack().mean()))
mx=-1;who=''
for n,L in LIB.items():
 z=pd.concat([F.stack(),L.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
 print('LIBRARY',n,'RHO',f(q),'CELLS',len(z))
 if np.isfinite(q) and abs(q)>mx:mx=abs(q);who=n
print('MAX_ABS_LIBRARY_CORRELATION',f(mx),'FACTOR',who)
