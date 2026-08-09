"""Miner 1: DXY persistent-trend sensitivity, validated as one macro-path factor."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; px={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); px[a]=pd.to_numeric(d.set_index('date').sort_index().close,errors='coerce')
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
d=get_index_daily_data('DXY',5000).copy();d.date=pd.to_datetime(d.date); dx=pd.to_numeric(d.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill(); dr=dx.pct_change()
# One idea: cross-asset sensitivity to a persistent (five-session) dollar path, rather than a one-day shock.
# At t, beta is a trailing 60-observation OLS slope of asset daily returns on contemporaneous DXY 5d return.
x=dr.rolling(5,min_periods=5).sum(); W=60
F=pd.DataFrame({a:R[a].rolling(W,min_periods=40).cov(x)/x.rolling(W,min_periods=40).var() for a in A})
print('FACTOR dxy_persistent_trend_sensitivity_60_5 cutoff',P.index.max().date(),'assets',len(A)); print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
def ev(h,span=None):
 f=F if span is None else F.loc[span[0]:span[1]]; y=(P.shift(-h).div(P)-1).reindex(f.index); z=[]; ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z)
 return dict(dates=len(z),IC=round(float(z.mean()),6),ICIR=round(float(z.mean()/z.std(ddof=1)),6),hit=round(float((z>0).mean()),4),avg_n=round(float(np.mean(ns)),2),min_n=int(np.min(ns)))
for h in [1,5,10,20]:print('H',h,ev(h))
for name,span in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027YTD',('2027-01-01','2028-01-01'))]:print('REGIME10',name,ev(10,span))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
# Exact-ish recreations of every admitted signal for novelty test.
med=R.median(axis=1); other={a:R.drop(columns=a).median(axis=1) for a in A};v20=R.rolling(20,min_periods=15).std();v5=R.rolling(5,min_periods=4).std(); trend=P.pct_change(20)/v20
S={'ravmom':trend,'volrev5':-P.pct_change(5)/v5,'idio':-R.sub(med,axis=0).rolling(20,min_periods=15).std(),'downbeta':pd.DataFrame({a:R[a].where(med<0).rolling(40,min_periods=20).cov(med.where(med<0))/med.where(med<0).rolling(40,min_periods=20).var() for a in A}),'skew':R.rolling(60,min_periods=40).skew(),'lagauto':-pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(R[a].shift(1)) for a in A}),'common':-pd.DataFrame({a:R[a].rolling(40,min_periods=25).corr(other[a]) for a in A}),'voltransition':-pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(R[a].shift(1)) for a in A})*np.log(v5/v20).clip(-2,2),'quiet':P.pct_change(20).abs()/R.abs().rolling(20,min_periods=15).sum()*(1-v20.rolling(60,min_periods=40).rank(pct=True)),'commonexp':pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(other[a]) for a in A}).rolling(20,min_periods=15).mean()-pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(other[a]) for a in A}).shift(20).rolling(20,min_periods=15).mean()}
vd=get_index_daily_data('VIX',5000).copy();vd.date=pd.to_datetime(vd.date);vr=pd.to_numeric(vd.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill().pct_change();S['vixtrend']=trend.mul(np.where(vr.rolling(20,min_periods=15).sum()>0,-1,1),axis=0);S['vixbeta']=-pd.DataFrame({a:R[a].where(vr>0).rolling(40,min_periods=15).cov(vr.where(vr>0))/vr.where(vr>0).rolling(40,min_periods=15).var() for a in A})
vols={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);vols[a]=pd.to_numeric(d.set_index('date').sort_index().get('volume'),errors='coerce').reindex(P.index)
q=pd.DataFrame(vols).replace(0,np.nan);lv=np.log(q/q.rolling(20,min_periods=15).mean());S['volume']=lv;S['stablevol']=-lv.rolling(20,min_periods=12).std();thr=med.rolling(60,min_periods=40).quantile(.35);S['downexcess']=R.sub(med,axis=0).where(med.shift(1)<thr.shift(1),axis=0).rolling(40,min_periods=10).median();S['tail']=-pd.DataFrame({a:(R[a]<R[a].rolling(60,min_periods=40).quantile(.2).shift(1)).rolling(40,min_periods=25).mean() for a in A});S['asym']=pd.DataFrame({a:R[a].where(med<0).rolling(60,min_periods=20).cov(med.where(med<0))/med.where(med<0).rolling(60,min_periods=20).var()-R[a].where(med>=0).rolling(60,min_periods=20).cov(med.where(med>=0))/med.where(med>=0).rolling(60,min_periods=20).var() for a in A})
# Existing DXY directional conditional mean asymmetry
up=(dr>0).astype(float);dn=(dr<0).astype(float);S['dxy_asym']=R.mul(dn,axis=0).rolling(60,min_periods=35).sum().div(dn.rolling(60,min_periods=35).sum(),axis=0)-R.mul(up,axis=0).rolling(60,min_periods=35).sum().div(up.rolling(60,min_periods=35).sum(),axis=0)
mx=0;who='';nmx=0
for name,g in S.items():
 z=pd.concat([F.rank(axis=1,pct=True).stack(),g.rank(axis=1,pct=True).stack()],axis=1).dropna();rho=float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);print('LIBCORR',name,'cells',len(z),'rho',round(rho,6))
 if abs(rho)>mx:mx=abs(rho);who=name;nmx=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',nmx,'N_FACTORS',len(S))
