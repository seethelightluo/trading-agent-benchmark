"""Miner 1: macro-volatility-conditioned peer-relationship stability (one factor idea)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list'];px={};vv={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index();px[a]=pd.to_numeric(d.close,errors='coerce');vv[a]=pd.to_numeric(d.get('volume'),errors='coerce')
P=pd.DataFrame(px).sort_index();R=P.pct_change();V=pd.DataFrame(vv).reindex(P.index);med=R.median(axis=1);other={a:R.drop(columns=a).median(axis=1) for a in A}
vd=get_index_daily_data('VIX',5000).copy();vd.date=pd.to_datetime(vd.date);vx=pd.to_numeric(vd.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill();vr=vx.pct_change()
# Stable correlation to the remaining-market median is favored only in an elevated VIX-volatility state.
c20=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(other[a]) for a in A}); base=-abs(c20-c20.rolling(60,min_periods=40).mean())
state=(vr.rolling(20,min_periods=15).std()/vr.rolling(60,min_periods=40).std()).clip(.5,2.0)
F=base.mul(state,axis=0)
print('FACTOR vix_conditioned_peer_relationship_stability_20_60 cutoff',P.index.max().date(),'assets',len(A));print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
def ev(h,span=None):
 f=F if span is None else F.loc[span[0]:span[1]];y=P.shift(-h).div(P)-1;z=[];ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   c=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(c):z.append(c);ns.append(len(q))
 z=np.array(z);return {'dates':len(z),'IC':round(float(z.mean()),6) if len(z) else None,'ICIR':round(float(z.mean()/z.std(ddof=1)),6) if len(z)>1 else None,'hit':round(float((z>0).mean()),4) if len(z) else None,'avg_n':round(float(np.mean(ns)),2) if ns else None,'min_n':min(ns) if ns else None}
for h in [1,5,10,20]:print('H',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(P.index.max().date())))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
# admitted-library reconstruction for mandatory pooled rank-correlation test
v20=R.rolling(20,min_periods=15).std();v5=R.rolling(5,min_periods=4).std();c40=pd.DataFrame({a:R[a].rolling(40,min_periods=25).corr(other[a]) for a in A})
S={'ravmom':P.pct_change(20)/v20,'volrev5':-P.pct_change(5)/v5,'idio':-R.sub(med,axis=0).rolling(20,min_periods=15).std(),'downbeta':pd.DataFrame({a:R[a].where(med<0).rolling(40,min_periods=20).cov(med.where(med<0))/med.where(med<0).rolling(40,min_periods=20).var() for a in A}),'skew':R.rolling(60,min_periods=40).skew(),'lagauto':-pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(R[a].shift(1)) for a in A}),'common':-c40,'voltransition':-pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(R[a].shift(1)) for a in A})*np.log(v5/v20).clip(-2,2),'quiet':P.pct_change(20).abs()/R.abs().rolling(20,min_periods=15).sum()*(1-v20.rolling(60,min_periods=40).rank(pct=True)),'commonexp':c40.rolling(20,min_periods=15).mean()-c40.shift(20).rolling(20,min_periods=15).mean(),'volume':np.log(V.replace(0,np.nan)/V.replace(0,np.nan).rolling(20,min_periods=15).mean()),'stablevol':-np.log(V.replace(0,np.nan)/V.replace(0,np.nan).rolling(20,min_periods=15).mean()).rolling(20,min_periods=12).std()}
S['vixtrend']=(P.pct_change(20)/v20).mul(np.where(vr.rolling(20,min_periods=15).sum()>0,-1,1),axis=0);S['vixbeta']=-pd.DataFrame({a:R[a].where(vr>0).rolling(40,min_periods=15).cov(vr.where(vr>0))/vr.where(vr>0).rolling(40,min_periods=15).var() for a in A});thr=med.rolling(60,min_periods=40).quantile(.35);S['downexcess']=R.sub(med,axis=0).where(med.shift(1)<thr.shift(1),axis=0).rolling(40,min_periods=10).median();S['tail']=-pd.DataFrame({a:(R[a]<R[a].rolling(60,min_periods=40).quantile(.2).shift(1)).rolling(40,min_periods=25).mean() for a in A});S['asym']=pd.DataFrame({a:R[a].where(med<0).rolling(60,min_periods=20).cov(med.where(med<0))/med.where(med<0).rolling(60,min_periods=20).var()-R[a].where(med>=0).rolling(60,min_periods=20).cov(med.where(med>=0))/med.where(med>=0).rolling(60,min_periods=20).var() for a in A});dd=get_index_daily_data('DXY',5000).copy();dd.date=pd.to_datetime(dd.date);dr=pd.to_numeric(dd.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill().pct_change();up=(dr>0).astype(float);dn=(dr<0).astype(float);S['dxy_asym']=R.mul(dn,axis=0).rolling(60,min_periods=35).sum().div(dn.rolling(60,min_periods=35).sum(),axis=0)-R.mul(up,axis=0).rolling(60,min_periods=35).sum().div(up.rolling(60,min_periods=35).sum(),axis=0);S['upconc']=-pd.DataFrame({a:(R[a]>0).astype(float).rolling(60,min_periods=40).mean() for a in A});S['downupbeta']=S['asym']
mx=0;who='';evidence=0
for n,g in S.items():
 q=pd.concat([F.rank(axis=1,pct=True).stack(),g.rank(axis=1,pct=True).stack()],axis=1).dropna();rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);print('LIBCORR',n,'cells',len(q),'rho',round(rho,6))
 if abs(rho)>mx:mx=abs(rho);who=n;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
