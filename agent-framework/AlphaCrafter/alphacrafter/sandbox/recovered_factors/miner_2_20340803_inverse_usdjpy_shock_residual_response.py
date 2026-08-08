"""Miner 2: contrarian USDJPY-shock residual response, inverse of prior persistence candidate."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2034-08-02')
def load(p): return pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].loc[:END]
px=pd.concat({a:load('../persistent/stock_data/'+a+'.csv') for a in A},axis=1).sort_index()
r=px.pct_change().replace([np.inf,-np.inf],np.nan); common=r.median(axis=1)
fx=load('../persistent/index_data/USDJPY.csv').pct_change().reindex(r.index)
# At t, estimate each beta and mean residual return on completed sessions whose preceding FX session was a top-30% absolute USDJPY shock; negate to state explicit mean-reversion hypothesis.
sig=pd.DataFrame(np.nan,index=r.index,columns=A)
for t in range(61,len(r)):
 ix=r.index[t-60:t]; f=fx.loc[ix]; response=(f.abs()>=f.abs().quantile(.70)).shift(1,fill_value=False); x=common.loc[ix]
 for a in A:
  y=r.loc[ix,a]; ok=y.notna()&x.notna()
  beta=np.cov(y[ok],x[ok],ddof=1)[0,1]/np.var(x[ok],ddof=1) if ok.sum()>=30 and np.var(x[ok])>0 else np.nan
  z=(y-beta*x)[response&y.notna()&x.notna()]
  if len(z)>=8: sig.iloc[t,sig.columns.get_loc(a)]=-z.mean()
def metrics(h):
 fwd=px.shift(-h)/px-1; v=[]; dates=[]; nn=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   c=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(c): v.append(c); dates.append(dt); nn.append(len(q))
 v=np.array(v); return v,pd.DatetimeIndex(dates),nn
print('FACTOR inverse-usdjpy-shock-residual-response-60obs endpoint',END.date())
print('cells',sig.notna().sum().sum(),'of',sig.size,'coverage',round(sig.notna().mean().mean(),6))
res={}
for h in (1,5,10,20):
 v,d,n=metrics(h);res[h]=(v,d);print('h',h,'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),6),'dates',len(v),'mean_n',round(np.mean(n),3))
for label,lo,hi in [('2026-2029','2026-01-01','2029-12-31'),('2030-2032','2030-01-01','2032-12-31'),('2033-end','2033-01-01',str(END.date()))]:
 v,d=res[10];z=v[(d>=lo)&(d<=hi)];print('regime10',label,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
r=sig.rank(axis=1,pct=True); ch=(r-r.shift()).abs().stack(); print('turnover',round(ch.mean(),6),'comparisons',len(ch),'median_iqr',round(sig.quantile(.75,axis=1).sub(sig.quantile(.25,axis=1)).median(),6))
sig.to_pickle('scripts/miner_2_20340803_inverse_usdjpy_shock_residual_response_signal.pkl')
