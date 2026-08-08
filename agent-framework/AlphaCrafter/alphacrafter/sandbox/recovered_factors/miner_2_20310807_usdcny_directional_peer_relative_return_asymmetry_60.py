"""One idea: USDCNY-directional peer-relative return asymmetry, with library novelty audit."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={};V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date');C[a]=pd.to_numeric(d.close,errors='coerce');V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C);V=pd.DataFrame(V).reindex(P.index);r=P.pct_change();m=r.median(axis=1);rel=r.sub(m,axis=0);cutoff=P.dropna(how='all').index.max()
def cs(x):return x.sub(x.median(axis=1),axis=0)
def ix(s):
 d=get_index_daily_data(s,5000).copy();d.date=pd.to_datetime(d.date);return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce').reindex(P.index)
# One interpretable idea: an asset's mean same-day peer-relative return on USDCNY appreciation minus depreciation days, trailing 60 completed sessions. Both states need 15 observations and signal is lagged.
cny=ix('USDCNY');cr=cny.pct_change();up=cr>0;down=cr<0
ok=(up.rolling(60,min_periods=40).sum()>=15)&(down.rolling(60,min_periods=40).sum()>=15)
raw=pd.DataFrame({a:rel[a].where(up).rolling(60,min_periods=15).mean()-rel[a].where(down).rolling(60,min_periods=15).mean() for a in A})
cand=cs(raw.where(ok,axis=0)).shift(1)
# Reconstruct admitted library factor signals from their operational definitions for mandatory novelty audit.
def beta(x,y,w,side=None):
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1)
 if side=='down':z=z.where(z.y<0)
 if side=='up':z=z.where(z.y>0)
 return z.x.rolling(w,min_periods=max(8,w//4)).cov(z.y)/z.y.rolling(w,min_periods=max(8,w//4)).var()
def eventmean(x,e,w,n):return x.where(e,axis=0).rolling(w,min_periods=n).mean()
vol20=r.rolling(20,min_periods=15).std();vol40=r.rolling(40,min_periods=15).std();vol5=r.rolling(5,min_periods=4).std();trend=P.pct_change(20)/vol20
other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A});corr20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(other[a]) for a in A})
vix=ix('VIX');vr=vix.pct_change();dxy=ix('DXY');dr=dxy.pct_change();jpy=ix('USDJPY').pct_change();y10=P.US10Y.pct_change()
neg=r.clip(upper=0);short=np.sqrt((neg*neg).rolling(10,min_periods=7).mean());long=np.sqrt((neg*neg).rolling(40,min_periods=25).mean());liq=np.log(V/V.rolling(20,min_periods=15).mean());pos=r.clip(lower=0)
me=m<m.rolling(60,min_periods=40).quantile(.35);q25=m.rolling(60,min_periods=40).quantile(.25);disp=r.std(axis=1);de=(disp>disp.rolling(60,min_periods=40).quantile(.75))&(disp>disp.shift(5));common=corr20.median(axis=1);ce=(common>common.rolling(60,min_periods=40).quantile(.75))&(common>common.shift(5))
def eb(x,y,e,w=60):
 z=pd.concat([x.rename('x'),y.rename('y'),e.rename('e')],axis=1).where(lambda z:z.e);return z.x.rolling(w,min_periods=12).cov(z.y)/z.y.rolling(w,min_periods=12).var()
S={
'downside_vol_compression':cs(-np.log((short+1e-5)/(long+1e-5))),'relative_volume_participation':liq,'upside_concentration':pos.rolling(60,min_periods=40).max()/pos.rolling(60,min_periods=40).sum(),'inverse_idio_vol':-rel.rolling(20,min_periods=15).std(),'risk_adjusted_trend':trend,'downside_correlation_spread':cs(pd.DataFrame({a:beta(r[a],m,20,'down')-beta(r[a],m,80,'down') for a in A})),'downside_beta_resilience':pd.DataFrame({a:beta(r[a],m,40,'down') for a in A}),'dxy_asymmetry':pd.DataFrame({a:r[a].where(dr<0).rolling(60,min_periods=15).mean()-r[a].where(dr>0).rolling(60,min_periods=15).mean() for a in A}),'vix_regime_trend':trend.mul(np.where(vix/vix.shift(20)-1>0,-1.,1.),axis=0),'stable_liquidity':-liq.rolling(20,min_periods=15).std(),'inverse_lower_tail_persistence':-pd.DataFrame({a:r[a].lt(r[a].rolling(60,min_periods=40).quantile(.2).shift(1)).rolling(40,min_periods=25).mean() for a in A}),'vix_upside_beta':pd.DataFrame({a:-beta(r[a],vr,40,'up') for a in A}),'volnorm_reversal':-P.pct_change(5)/vol5,'return_skewness':r.rolling(60,min_periods=40).skew(),'serial_dependence':cs(-pd.DataFrame({a:rel[a].rolling(20,min_periods=16).corr(rel[a].shift(1)) for a in A})),'yield_shock_resilience':cs(pd.DataFrame({a:eb(r[a],y10,y10.abs()>y10.abs().rolling(60,min_periods=40).quantile(.75))-beta(r[a],y10,60) for a in A})),'yield_vol_transition':cs(pd.DataFrame({a:eb(r[a],y10,y10.abs().rolling(20,min_periods=15).mean()>y10.abs().rolling(60,min_periods=40).mean())-beta(r[a],y10,60) for a in A})),'jpy_vol_transition':cs(pd.DataFrame({a:eb(r[a],jpy,jpy.abs().rolling(20,min_periods=15).mean()>jpy.abs().rolling(60,min_periods=40).mean())-beta(r[a],jpy,60) for a in A})),'dispersion_reversal':cs(-eventmean(rel.rolling(5,min_periods=5).sum().div(vol20),de,20,5)),'commonality_reversal':cs(-eventmean(P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0).div(vol20),ce,60,1))}
fw={h:P.shift(-h)/P-1 for h in(1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];z=[];n=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fw[h].reindex(x.index).loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);n.append(len(q))
 z=np.array(z)
 return dict(dates=len(z),ic=round(z.mean(),6),icir=round(z.mean()/z.std(ddof=1),6),hit=round((z>0).mean(),6),breadth=round(np.mean(n),3),min_breadth=min(n)) if len(z) else {'dates':0}
print('FACTOR usdcny_directional_peer_relative_return_asymmetry_60 CUTOFF',cutoff.date(),'ASSETS',len(A));print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(cand.std(axis=1).mean(),6))
for h in(1,5,10,20):print('H',h,stat(h))
for name,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',name,stat(10,p))
mx=0;who='';evidence=0
for name,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 if abs(rho)>mx:mx=abs(rho);who=name;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS_AUDITED',len(S))
print('NOTE: audit reconstructs operational representatives; admission requires evidence against every persisted admitted signal.')
