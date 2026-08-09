"""miner_1: trailing 20-session cross-asset-market residual trend; data API is cursor-safe."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; H=[1,5,10,20]
px={}; vv={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
 px[a]=pd.to_numeric(d['close'],errors='coerce'); vv[a]=pd.to_numeric(d.get('volume'),errors='coerce').replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vv).reindex(P.index); R=P.pct_change(fill_method=None); M=R.median(axis=1)
# One interpretable idea: 20d cumulative return unexplained by concurrent broad cross-asset return.
# beta is trailing and shifted one day, preventing the current return from setting its own exposure.
beta=R.rolling(60,min_periods=40).cov(M).unstack().reindex(columns=A).div(M.rolling(60,min_periods=40).var(),axis=0).shift(1)
resid=R-beta.mul(M,axis=0)
F=resid.rolling(20,min_periods=15).sum()
trend=P.pct_change(20,fill_method=None)/R.rolling(20,min_periods=15).std()
rev5=-P.pct_change(5,fill_method=None)/R.rolling(5,min_periods=4).std(); rev1=-R/R.rolling(20,min_periods=15).std()
rv=np.log(V/V.rolling(20,min_periods=15).mean()); eff=P.pct_change(20,fill_method=None).abs()/R.abs().rolling(20,min_periods=15).sum(); vp=R.rolling(20,min_periods=15).std().rolling(60,min_periods=40).rank(pct=True)
auto=-R.rolling(20,min_periods=15).apply(lambda x:x.dropna().autocorr(1) if len(x.dropna())>=15 else np.nan,raw=False)
LIB={'risk_adjusted_trend_20d':trend,'ravmom_20obs':trend,'volnorm_reversal_5obs':rev5,'volscaled_reversal_1obs':rev1,'relative_volume_participation_20d':rv,'quiet_trend_path_efficiency_20_60':eff*(1-vp),'inverse_lag1_return_autocorrelation_20':auto}
try:
 d=get_index_daily_data('VIX',5000).copy();d['date']=pd.to_datetime(d.date);v=pd.to_numeric(d.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill();vu=v.pct_change()
 LIB['vix_regime_conditioned_risk_adjusted_trend_20']=trend.where(v.pct_change(20)<=0,-trend)
 LIB['vix_upside_shock_beta_resilience_40']=-R.where(vu>0).rolling(40,min_periods=12).cov(vu.where(vu>0)).div(vu.where(vu>0).rolling(40,min_periods=12).var(),axis=0)
except Exception as e: print('VIX error',e)
# Existing downside cross-asset beta.
neg=M<0; LIB['downside_cross_asset_beta_resilience_40']=R.where(neg).rolling(40,min_periods=12).cov(M.where(neg)).div(M.where(neg).rolling(40,min_periods=12).var(),axis=0)
def stat(h,lo=None,hi=None):
 fw=P.shift(-h).div(P)-1; dates=F.index if lo is None else F.loc[lo:hi].index; z=[]; n=[]
 for t in dates:
  q=pd.concat([F.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   x=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(x):z.append(x);n.append(len(q))
 z=np.array(z); return (len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)),float((z>0).mean()),float(np.mean(n)),int(np.min(n))) if len(z)>1 else (0,np.nan,np.nan,np.nan,np.nan,0)
print('FACTOR residual_cross_asset_trend_20_60 END',P.index.max().date());print('CELLS',int(F.notna().sum().sum()),'/',F.size,'COVERAGE',float(F.notna().stack().mean()))
for h in H: print('H',h,'dates IC ICIR hit meanN minN',stat(h))
for label,lo,hi in [('2020','2020-01-01','2020-12-31'),('2021_22','2021-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027YTD','2027-01-01',str(P.index.max().date()))]:print('REGIME10',label,stat(10,lo,hi))
print('TURNOVER',float(F.rank(axis=1,pct=True).diff().abs().stack().mean()))
mx=0; who=''
for k,x in LIB.items():
 q=pd.concat([F.stack(),x.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);print('LIBCORR',k,len(q),rho)
 if abs(rho)>mx:mx=abs(rho);who=k
print('MAX_ABS_LIBRARY_CORRELATION',mx,who)
