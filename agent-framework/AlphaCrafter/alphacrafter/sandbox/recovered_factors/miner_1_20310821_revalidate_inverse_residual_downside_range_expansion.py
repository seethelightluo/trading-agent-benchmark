import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2031-08-20')
D={}
for a in AS:
 x=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index(); D[a]=x.loc[x.index<=CUT]
idx=sorted(set.intersection(*[set(x.index) for x in D.values()])); close=pd.DataFrame({a:D[a].reindex(idx).close for a in AS}); hi=pd.DataFrame({a:D[a].reindex(idx).high for a in AS}); lo=pd.DataFrame({a:D[a].reindex(idx).low for a in AS})
r=close.pct_change(); med=r.median(axis=1); beta=r.rolling(60,min_periods=45).cov(med).div(med.rolling(60,min_periods=45).var(),axis=0); resid=r-beta.mul(med,axis=0)
rng=(hi-lo).div(close.shift(1)); norm=rng/rng.rolling(20,min_periods=12).median(); sig=-norm.where(resid.shift(1)<0).rolling(60,min_periods=12).mean()
# ICs
def ics(h):
 fwd=close.shift(-h)/close-1; out=[]; ns=[]
 for t in idx:
  z=pd.concat([sig.loc[t],fwd.loc[t]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 return np.array(out),np.array(ns)
print('cutoff',CUT.date(),'calendar common dates',len(idx),'signal cells',int(sig.notna().sum().sum()),'/',sig.size)
for h in [1,5,10,20]:
 x,n=ics(h); print('H',h,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6),'meanN',round(n.mean(),3))
# selected h per era
fwd=close.shift(-20)/close-1
for name,start,end in [('2026_2029','2026-01-01','2029-12-31'),('2030_current','2030-01-01','2031-08-20')]:
 ar=[]
 for t in idx:
  if not(pd.Timestamp(start)<=t<=pd.Timestamp(end)):continue
  z=pd.concat([sig.loc[t],fwd.loc[t]],axis=1).dropna()
  if len(z)>=8:ar.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 ar=np.array(ar);print('REGIME',name,'dates',len(ar),'IC',round(ar.mean(),6),'ICIR',round(ar.mean()/ar.std(ddof=1),6),'hit',round((ar>0).mean(),6))
# turnover, distribution
pairs=[]
for i in range(1,len(idx)):
 a=sig.iloc[i-1];b=sig.iloc[i];z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:pairs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('turnover',round(1-np.mean(pairs),6),'pairs',len(pairs),'medianIQR',round(sig.quantile(.75,axis=1).sub(sig.quantile(.25,axis=1)).median(),6))
# Full current-signals stored unavailable generically: current factor vs risk trend explicitly, historical original novelty evidence preserved in record
trend=(close/close.shift(20)-1).div(r.rolling(20).std())
cs=[]
for t in idx:
 z=pd.concat([sig.loc[t],trend.loc[t]],axis=1).dropna()
 if len(z)>=8:cs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('diagnostic corr to risk_adjusted_trend maxabs',round(np.max(np.abs(cs)),6),'cells dates',len(cs))
PY
python scripts/miner_1_20310821_revalidate_inverse_residual_downside_range_expansion.py