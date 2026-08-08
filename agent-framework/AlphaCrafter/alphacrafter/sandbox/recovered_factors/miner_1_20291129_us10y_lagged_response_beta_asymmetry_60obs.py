"""miner_1: asymmetry of lagged US10Y transmission, one interpretable factor."""
import numpy as np, pandas as pd, glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-11-28'); ROOT='../persistent/stock_data'
D={a:pd.read_csv(f'{ROOT}/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END] for a in A}
ix=pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in D.values()])))
def fld(k): return pd.DataFrame({a:D[a][k].reindex(ix).astype(float) for a in A})
c=fld('close'); r=c.pct_change(fill_method=None); med=r.median(axis=1); MM=pd.DataFrame({a:med for a in A}); s20=r.rolling(20,min_periods=15).std()
def macro(n):
 x=pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().iloc[:,0].astype(float).reindex(ix)
 return x.pct_change(fill_method=None)
def beta(x,z,w=60): return x.rolling(w,min_periods=15).cov(z).div(z.rolling(w,min_periods=15).var(),axis=0)
def corr(x,z,w=60): return x.rolling(w,min_periods=15).corr(z)
dxy=macro('DXY'); vix=macro('VIX'); y=r.US10Y.shift(1)
# Difference in asset sensitivities to prior-day positive versus negative rate changes.
f=beta(r.where(y>0),y.where(y>0))-beta(r.where(y<0),y.where(y<0))
b=beta(r,MM); resid=r.sub(b*MM,axis=0); disp=r.std(axis=1); rng=(fld('high')-fld('low')).div(c)
lib={'ravmom':(c/c.shift(20)-1)/s20,'volnorm_reversal':-(c/c.shift(5)-1)/r.rolling(5,min_periods=4).std(),'correlation_asymmetry':corr(r.where(MM<0),MM.where(med<0))-corr(r.where(MM>=0),MM.where(med>=0)),'return_sign_balance':r.gt(0).rolling(20,min_periods=15).mean()-.5,'dispersion_sensitivity':corr(r,pd.DataFrame({a:disp for a in A}),20),'volatility_clustering':r.abs().rolling(20,min_periods=15).corr(r.abs().shift(1)),'vix_shock_relief':beta(r.where(vix>0),vix.where(vix>0))-beta(r.where(vix<0),vix.where(vix<0)),'dxy_median_trend':beta(r.where(med.rolling(20,min_periods=15).median()>0),dxy.where(med.rolling(20,min_periods=15).median()>0))-beta(r.where(med.rolling(20,min_periods=15).median()<=0),dxy.where(med.rolling(20,min_periods=15).median()<=0)),'dxy_relvol':beta(r,dxy.rolling(20,min_periods=15).std())-beta(r,dxy.rolling(60,min_periods=15).std()),'vix_tail_lag':beta(r,vix.shift(1).where(vix.shift(1).abs()>vix.abs().rolling(60,min_periods=30).quantile(.8))),'vol_orth_beta':b.sub(b.median(axis=1),axis=0),'resid_downsemi':-resid.where(resid<0).pow(2).rolling(60,min_periods=15).mean().div(resid.pow(2).rolling(60,min_periods=15).mean()),'return_persistence':r.rolling(20,min_periods=15).corr(r.shift(1)),'inv_downside_volume_accel':None,'inverted_range_state':-np.log(rng.rolling(20,min_periods=15).mean()/rng.rolling(60,min_periods=15).mean()),'adaptive_vix':beta(r.where(vix<0),vix.where(vix<0),25)-beta(r.where(vix<0),vix.where(vix<0),60),'dxy_shocklag':beta(r,dxy.shift(1).where(dxy.shift(1).abs()>dxy.abs().rolling(60,min_periods=30).quantile(.8))),'excess_downside_beta':beta(r.where(MM<0),MM.where(med<0))-b,'realized_vol':s20}
# complete library volume acceleration reconstruction
v=fld('volume').replace(0,np.nan); lib['inv_downside_volume_accel']=-np.log(v.where(r<0).rolling(20,min_periods=8).mean()/v.where(r>=0).rolling(20,min_periods=8).mean())+np.log(v.where(r<0).rolling(60,min_periods=12).mean()/v.where(r>=0).rolling(60,min_periods=12).mean())
print('FACTOR us10y_lagged_response_beta_asymmetry_60obs endpoint',END.date(),'assets',len(A));print('cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),5))
def getic(h):
 out=[]; ds=[]; ns=[]
 for t in range(len(f)-h):
  q=pd.concat([f.iloc[t],c.iloc[t+h]/c.iloc[t]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: out.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(f.index[t]);ns.append(len(q))
 return np.array(out),pd.DatetimeIndex(ds),np.array(ns)
res=[]
for h in [1,5,10,20]:
 x,ds,ns=getic(h); av=x.mean(); ir=av/x.std(ddof=1);res.append((abs(av*ir),h,av,ir,x,ds,ns));print('H',h,'dates',len(x),'IC',round(av,6),'ICIR',round(ir,6),'hit',round((x>0).mean(),5),'mean_instruments',round(ns.mean(),3))
_,h,av,ir,x,ds,ns=max(res,key=lambda z:z[0]); print('SELECTED',h)
for name,lo,hi in [('2020_21','2020-01-01','2021-12-31'),('2022_23','2022-01-01','2023-12-31'),('2024_25','2024-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31')]:
 z=x[(ds>=lo)&(ds<=hi)];print('REGIME',name,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,'hit',round((z>0).mean(),5) if len(z) else None)
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('rank_turnover',round(np.mean(turns),6))
mx=-1; who=''; evidence=0
for name,g in lib.items():
 vals=[]
 for t in f.index:
  q=pd.concat([f.loc[t],g.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 if vals and max(map(abs,vals))>mx: mx=max(map(abs,vals));who=name;evidence=len(vals)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'WITH',who,'EVIDENCE_DATES',evidence,'library_json_count',len(glob.glob('factors/*.json')))
print('ADMISSION',abs(av)>=.007 and abs(ir)>=.084 and mx<.5)
