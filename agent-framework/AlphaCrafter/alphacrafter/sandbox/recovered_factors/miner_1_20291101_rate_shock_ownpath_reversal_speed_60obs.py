"""One candidate: own-path rate-tail reversal-speed asymmetry, 60 observations.
For every asset, compare its mean response one session after a large US10Y move
with its delayed (3--5 session) response to those shocks. This measures how
quickly the asset absorbs or reverses a rate shock, without common risk-state partitioning.
"""
import numpy as np,pandas as pd,glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2029-10-31'; ROOT='../persistent/stock_data'
D={a:pd.read_csv(f'{ROOT}/{a}.csv').set_index('date').sort_index().loc[:END] for a in A}; ix=sorted(set().union(*[set(x.index) for x in D.values()]))
def fld(k):return pd.DataFrame({a:D[a].reindex(ix)[k] for a in A})
c=fld('close');op=fld('open');hi=fld('high');lo=fld('low');v=fld('volume').replace(0,np.nan);r=c.pct_change(fill_method=None);med=r.median(1);MM=pd.DataFrame({a:med for a in A});disp=r.std(1);s20=r.rolling(20,min_periods=15).std()
def macro(n):return pd.read_csv('../persistent/index_data/'+n+'.csv').set_index('date').sort_index().reindex(ix).iloc[:,0].pct_change(fill_method=None)
vix=macro('VIX');dxy=macro('DXY')
def beta(x,z,w=60):return x.rolling(w,min_periods=15).cov(z).div(z.rolling(w,min_periods=15).var(),axis=0)
def corr(x,z,w=60):return x.rolling(w,min_periods=15).corr(z)
b=beta(r,MM);resid=r.sub(b*MM,axis=0);neg=MM.where(med<0);pos=MM.where(med>=0);range_=(hi-lo)/c
# Large rate shocks are trailing 80th-percentile absolute US10Y changes. Difference
# between immediate response (shock at t-1) and delayed response (t-3 through t-5).
us=c.US10Y.ffill().pct_change(fill_method=None); tail=us.abs()>us.abs().rolling(60,min_periods=30).quantile(.8)
imm=r.where(tail.shift(1),np.nan).rolling(60,min_periods=12).mean()
delayed=(r.where(tail.shift(3),np.nan)+r.where(tail.shift(4),np.nan)+r.where(tail.shift(5),np.nan)).rolling(60,min_periods=12).mean()/3
f=imm-delayed
lib={'ravmom':(c/c.shift(20)-1)/s20,'volnorm_reversal':-(c/c.shift(5)-1)/r.rolling(5,min_periods=4).std(),'correlation_asymmetry':corr(r.where(MM<0),neg)-corr(r.where(MM>=0),pos),'return_sign_balance':r.gt(0).rolling(20,min_periods=15).mean()-.5,'dispersion_sensitivity':corr(r,disp,20),'volatility_clustering':r.abs().rolling(20,min_periods=15).corr(r.abs().shift(1)),'overnight_daytime':-((op/c.shift(1)-1)*(c/op-1)).rolling(20,min_periods=15).mean(),'vix_shock_relief':beta(r.where(vix>0),vix.where(vix>0))-beta(r.where(vix<0),vix.where(vix<0)),'dxy_median_trend':beta(r.where(med.rolling(20,min_periods=15).median()>0),dxy.where(med.rolling(20,min_periods=15).median()>0))-beta(r.where(med.rolling(20,min_periods=15).median()<=0),dxy.where(med.rolling(20,min_periods=15).median()<=0)),'dxy_relvol':beta(r,dxy.rolling(20,min_periods=15).std())-beta(r,dxy.rolling(60,min_periods=20).std()),'vix_tail_lag':beta(r,vix.shift(1).where(vix.shift(1).abs()>vix.abs().rolling(60,min_periods=30).quantile(.8))),'vol_orth_beta':b.sub(b.median(1),axis=0),'resid_downsemi':-resid.where(resid<0).pow(2).rolling(60,min_periods=15).mean()/resid.pow(2).rolling(60,min_periods=15).mean(),'return_persistence':r.rolling(20,min_periods=15).corr(r.shift(1)),'direction_eff':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),'liquidity_stress':np.log((r.abs()/v).rolling(20,min_periods=15).mean()/(r.abs()/v).rolling(60,min_periods=15).mean()),'downside_volume':np.log(v.where(r<0).rolling(60,min_periods=12).mean()/v.where(r>=0).rolling(60,min_periods=12).mean()),'inv_downside_volume_accel':-np.log(v.where(r<0).rolling(20,min_periods=8).mean()/v.where(r>=0).rolling(20,min_periods=8).mean())+np.log(v.where(r<0).rolling(60,min_periods=12).mean()/v.where(r>=0).rolling(60,min_periods=12).mean()),'inverted_range_state':-np.log(range_.rolling(20,min_periods=15).mean()/range_.rolling(60,min_periods=15).mean()),'adaptive_vix':beta(r.where(vix<0),vix.where(vix<0),25)-beta(r.where(vix<0),vix.where(vix<0),60),'dxy_shocklag':beta(r,dxy.shift(1).where(dxy.shift(1).abs()>dxy.abs().rolling(60,min_periods=30).quantile(.8))),'excess_downside_beta':beta(r.where(MM<0),neg)-b,'realized_vol':s20}
vis=c.index[c.index<=END]
def stats(sub,h):
 fw=c.shift(-h)/c-1;zic=[];nn=[];to=[];prev=None
 for t in sub:
  z=pd.concat([f.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:zic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);nn.append(len(z))
  q=f.loc[t].rank();z=pd.concat([q,prev],axis=1).dropna() if prev is not None else pd.DataFrame()
  if len(z)>=8:to.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  prev=q
 x=np.array(zic);return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(nn),np.mean(to)
print('FACTOR rate_shock_ownpath_reversal_speed_60obs','END',END,'assets',len(A),'cells',f.loc[vis].notna().sum().sum(),'of',len(vis)*15)
for h in [1,5,10,20]:print('H',h,tuple(round(y,6) if isinstance(y,float) else y for y in stats(vis,h)))
for n,s in [('2020_21',vis[vis<'2022-01-01']),('2022_23',vis[(vis>='2022-01-01')&(vis<'2024-01-01')]),('2024_25',vis[(vis>='2024-01-01')&(vis<'2026-01-01')]),('2026_current',vis[vis>='2026-01-01'])]:print('REGIME',n,tuple(round(y,6) if isinstance(y,float) else y for y in stats(s,20)[:4]))
mx=-1;who='';evid=0
for n,x in lib.items():
 q=[]
 for t in vis:
  z=pd.concat([f.loc[t],x.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 m=max(map(abs,q)) if q else np.nan
 if np.isfinite(m) and m>mx:mx,who,evid=m,n,len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'WITH',who,'EVIDENCE_DATES',evid,'library_json_count',len(glob.glob('factors/*.json')))
