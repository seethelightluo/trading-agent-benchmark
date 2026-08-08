"""One idea: DXY-volatility-transition beta resilience; macro currency turbulence sensitivity."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={};V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date');C[a]=pd.to_numeric(d.close,errors='coerce');V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C);V=pd.DataFrame(V).reindex(P.index);r=P.pct_change();m=r.median(axis=1);rel=r.sub(m,axis=0);cutoff=P.dropna(how='all').index.max()
def cs(x):return x.sub(x.median(axis=1),axis=0)
def beta(x,y,w=60):
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1);return z.x.rolling(w,min_periods=15).cov(z.y)/z.y.rolling(w,min_periods=15).var()
def ix(s):
 d=get_index_daily_data(s,5000).copy();d.date=pd.to_datetime(d.date);return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce').reindex(P.index)
dxy=ix('DXY');dxr=dxy.pct_change();dxv=dxr.rolling(10,min_periods=8).std();event=(dxv>dxv.rolling(60,min_periods=40).quantile(.75))&(dxv>dxv.shift(5))
def eb(x,y,e):
 z=pd.concat([x.rename('x'),y.rename('y'),e.rename('e')],axis=1);z[['x','y']]=z[['x','y']].where(z.e,axis=0);return z.x.rolling(60,min_periods=15).cov(z.y)/z.y.rolling(60,min_periods=15).var()
cand=cs(-pd.DataFrame({a:eb(r[a],dxr,event) for a in A})+pd.DataFrame({a:beta(r[a],dxr) for a in A})).shift(1)
# Load persisted definitions' expressions is insufficient to recreate signals.  Recreate broad existing signal families for conservative correlation audit.
vol20=r.rolling(20,min_periods=15).std(); trend=P.pct_change(20)/vol20; other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A});corr20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(other[a]) for a in A}); y=r.US10Y; vix=ix('VIX');vr=vix.pct_change(); q=m.rolling(60,min_periods=40).quantile(.25);liq=np.log(V/V.rolling(20,min_periods=15).mean()); neg=r.clip(upper=0); short=np.sqrt((neg*neg).rolling(10,min_periods=7).mean());long=np.sqrt((neg*neg).rolling(40,min_periods=25).mean())
def downbeta(x,y,w):
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1).where(lambda z:z.y<0);return z.x.rolling(w,min_periods=15).cov(z.y)/z.y.rolling(w,min_periods=15).var()
def evmean(x,e,w,n):return x.where(e,axis=0).rolling(w,min_periods=n).mean()
S={
'inverse_idiosyncratic_volatility_20':-rel.rolling(20,min_periods=15).std(),'risk_adjusted_trend_20d':trend,'ravmom_20obs':trend,'volnorm_reversal_5obs':-P.pct_change(5)/r.rolling(5,min_periods=4).std(),'volscaled_reversal_1obs':-r/vol20,'stable_liquidity_participation_20':-liq.rolling(20,min_periods=15).std(),'relative_volume_participation_20d':liq,'return_skewness_60':r.rolling(60,min_periods=40).skew(),'inverse_lower_tail_persistence_40_60':-pd.DataFrame({a:r[a].lt(r[a].rolling(60,min_periods=40).quantile(.2).shift()).rolling(40,min_periods=25).mean() for a in A}),'dxy_directional_return_asymmetry_60':pd.DataFrame({a:r[a].where(dxr<0).rolling(60,min_periods=15).mean()-r[a].where(dxr>0).rolling(60,min_periods=15).mean() for a in A}),'vix_upside_shock_beta_resilience_40':pd.DataFrame({a:-beta(r[a],vr,40) for a in A}),'downside_cross_asset_beta_resilience_40':pd.DataFrame({a:downbeta(r[a],m,40) for a in A}),'peer_relative_downside_volatility_compression_10_40':cs(-np.log((short+1e-5)/(long+1e-5))),'gradual_volatility_contraction_gated_trend_20':trend*np.tanh((-np.log(vol20/r.rolling(40,min_periods=15).std())).clip(-2,2)),'downside_correlation_regime_spread_20_80':cs(pd.DataFrame({a:downbeta(r[a],m,20)-downbeta(r[a],m,80) for a in A}))}
fw={h:P.shift(-h)/P-1 for h in(1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1];z=[];breadth=[]
 for d in x.index:
  qx=pd.concat([x.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(qx)>=8:z.append(spearmanr(qx.iloc[:,0],qx.iloc[:,1]).statistic);breadth.append(len(qx))
 z=np.array(z);return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),6),'breadth':round(np.mean(breadth),2),'min_breadth':min(breadth)}
print('FACTOR dxy_volatility_transition_beta_resilience_60 CUTOFF',cutoff.date(),'ASSETS',len(A));print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(cand.std(axis=1).mean(),6))
for h in(1,5,10,20):print('H',h,stats(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stats(10,p))
mx=-1
for n,g in S.items():
 z=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
 print('LIBCORR',n,'cells',len(z),'rho',round(rho,6))
 if abs(rho)>mx:mx=abs(rho);who=n;ev=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
