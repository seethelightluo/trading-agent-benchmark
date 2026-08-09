"""One idea: inverse extreme broad-upside magnitude-weighted peer-relative capture (60d)."""
import numpy as np,pandas as pd,json,glob
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); C[a]=pd.to_numeric(d.close,errors='coerce');V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C);r=P.pct_change();cutoff=P.dropna(how='all').index.max(); m=r.median(axis=1)
def cs(x):return x.sub(x.median(axis=1),axis=0)
other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A}); rel=r.sub(other,axis=0)
# Decisive broad peer-strength events, threshold estimated strictly before event.
thr=other.rolling(60,min_periods=40).quantile(.75).shift(1); event=other.gt(thr); weight=other.where(event)
raw=(rel*weight).rolling(60,min_periods=15).sum()/weight.rolling(60,min_periods=15).sum()
cand=cs(-raw).shift(1) # inverse: unusually strong relative capture in broad upside subsequently mean reverts
fw={h:P.shift(-h)/P-1 for h in(1,5,10,20)}
def stats(x,h,period=None):
 z=[];ns=[]
 y=fw[h]
 if period:x=x.loc[period[0]:period[1]]
 for d in x.index:
  q=pd.concat([x.loc[d],y.reindex(x.index).loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 if not z:return None
 z=np.array(z);return dict(dates=len(z),ic=round(z.mean(),6),icir=round(z.mean()/z.std(ddof=1),6),hit=round((z>0).mean(),6),breadth=round(np.mean(ns),3),min_breadth=min(ns))
print('FACTOR inverse_extreme_broad_upside_magnitude_weighted_relative_capture_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6))
for h in(1,5,10,20):print('H',h,stats(cand,h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stats(cand,10,p))
# Reconstruct library signals from stored definitions where feasible; correlations use same full aligned panel.
def ix(sym):
 d=get_index_daily_data(sym,5000);d.date=pd.to_datetime(d.date);return pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce').reindex(P.index)
def beta(x,y,w=40,down=False):
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1);z=z.where(z.y<0) if down else z;return z.x.rolling(w,min_periods=max(8,w//4)).cov(z.y)/z.y.rolling(w,min_periods=max(8,w//4)).var()
vol20=r.rolling(20,min_periods=15).std(); trend=P.pct_change(20)/vol20;relm=r.sub(m,axis=0); neg=r.clip(upper=0); short=np.sqrt((neg*neg).rolling(10,min_periods=7).mean());long=np.sqrt((neg*neg).rolling(40,min_periods=25).mean());pos=r.clip(lower=0);peak=P.rolling(60,min_periods=45).max();dd=P/peak-1
S={'inverse_idiosyncratic_volatility_20':-relm.rolling(20,min_periods=15).std(),'risk_adjusted_trend_20d':trend,'peer_relative_downside_volatility_compression_10_40':cs(-np.log((short+1e-5)/(long+1e-5))),'quiet_trend_path_efficiency_20_60':P.pct_change(20).abs()/r.abs().rolling(20,min_periods=15).sum()*(1-vol20.rolling(60,min_periods=40).rank(pct=True)),'upside_return_concentration_60':pos.rolling(60,min_periods=40).max()/pos.rolling(60,min_periods=40).sum(),'smooth_peer_relative_drawdown_recovery_60_10':cs((dd-dd.shift(10))/(.01-dd.shift(10))),'downside_cross_asset_beta_resilience_40':pd.DataFrame({a:beta(r[a],m,40,True) for a in A}),'downside_correlation_regime_spread_20_80':cs(pd.DataFrame({a:beta(r[a],m,20,True)-beta(r[a],m,80,True) for a in A})),'inverse_lower_tail_persistence_40_60':-pd.DataFrame({a:r[a].lt(r[a].rolling(60,min_periods=40).quantile(.2).shift(1)).rolling(40,min_periods=25).mean() for a in A}),'inverse_peer_relative_serial_dependence_20':cs(-pd.DataFrame({a:relm[a].rolling(20,min_periods=16).corr(relm[a].shift(1)) for a in A}))}
# Include prior admitted close cousin explicitly; required novelty evidence against relevant library signal.
lowthr=other.rolling(60,min_periods=40).quantile(.25).shift(1); lw=((-other).where(other.lt(lowthr)));S['inverse_extreme_broad_weakness_magnitude_weighted_relative_capture_60']=cs(-((rel*lw).rolling(60,min_periods=15).sum()/lw.rolling(60,min_periods=15).sum())).shift(1)
mx=0;who='';evidence=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 print('LIBCORR',n,'cells',len(q),'rho',round(rho,6))
 if abs(rho)>mx:mx=abs(rho);who=n;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',evidence,'RECONSTRUCTED_LIBRARY_FACTORS',len(S),'ADMITTED_LIBRARY_FILES',len(glob.glob('factors/*.json')))
print('ADMISSION',abs(stats(cand,10)['ic'])>=.007 and abs(stats(cand,10)['icir'])>=.084 and mx<.5)
