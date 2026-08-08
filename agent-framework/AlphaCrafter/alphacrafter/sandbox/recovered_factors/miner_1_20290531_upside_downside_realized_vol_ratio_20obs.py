import numpy as np, pandas as pd
from scipy.stats import spearmanr
# Candidate: upside/downside realized-volatility asymmetry. A high value means recent
# positive return variability dominates negative variability, a broad path-recovery measure.
ROOT='../persistent/stock_data'; A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2029-05-30'
D={a:pd.read_csv(f'{ROOT}/{a}.csv').set_index('date').sort_index().loc[:END] for a in A}; ix=sorted(set().union(*[set(d.index) for d in D.values()]))
c=pd.DataFrame({a:D[a].reindex(ix)['close'] for a in A}); r=c.pct_change(fill_method=None); vis=c.index[c.index<=END]
up=r.clip(lower=0).pow(2).rolling(20,min_periods=15).mean().pow(.5); down=(-r.clip(upper=0)).pow(2).rolling(20,min_periods=15).mean().pow(.5)
f=np.log((up+1e-8)/(down+1e-8)).clip(-5,5)
print('FACTOR upside_downside_realized_vol_ratio_20obs endpoint',vis[-1],'assets',len(A),'cells',int(f.loc[vis].notna().sum().sum()),'of',len(vis)*len(A))
def stat(sub,h):
 fw=c.shift(-h).div(c)-1; vals=[];ns=[];turn=[];prev=None
 for t in sub:
  z=pd.concat([f.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  q=f.loc[t].rank(); z2=pd.concat([q,prev],axis=1).dropna() if prev is not None else pd.DataFrame()
  if len(z2)>=8: turn.append(1-spearmanr(z2.iloc[:,0],z2.iloc[:,1]).statistic)
  prev=q
 x=np.array(vals); return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns),np.mean(turn)
for h in [1,5,10,20]:
 x=stat(vis,h);print('H',h,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4),'mean_n',round(x[4],2),'coverage',round(f.loc[vis].notna().mean().mean(),4),'turn',round(x[5],4))
for label,sub in [('2020_21',vis[vis<'2022-01-01']),('2022_23',vis[(vis>='2022-01-01')&(vis<'2024-01-01')]),('2024_25',vis[(vis>='2024-01-01')&(vis<'2026-01-01')]),('2026_current',vis[vis>='2026-01-01'])]:
 x=stat(sub,5);print('REGIME',label,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4))
# diversification pre-screen (candidate must clear performance first; exact full-library reconstruction is only needed if it does).
med=r.median(axis=1); MM=pd.DataFrame({a:med for a in A});disp=r.std(axis=1)
def beta(x,y,w=60):return x.rolling(w,min_periods=12).cov(y).div(y.rolling(w,min_periods=12).var(),axis=0)
def co(x,y,w=60):return x.rolling(w,min_periods=12).corr(y)
lib={'risk_adjusted_trend':(c/c.shift(20)-1)/r.rolling(20,min_periods=15).std(),'return_persistence':r.rolling(20,min_periods=15).corr(r.shift(1)),'inverse_volatility_clustering':-r.abs().rolling(20,min_periods=15).corr(r.abs().shift(1)),'return_sign_balance':r.gt(0).rolling(20,min_periods=15).mean()-.5,'directional_efficiency':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),'vol_orthogonal_median_beta':beta(r,MM).sub(beta(r,MM).median(axis=1),axis=0),'inverse_dispersion_sensitivity':-r.rolling(20,min_periods=15).corr(disp)}
mx=0
for n,x in lib.items():
 q=[]
 for t in vis:
  z=pd.concat([f.loc[t],x.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 m=max(map(abs,q)) if q else np.nan;mx=max(mx,m) if np.isfinite(m) else mx;print('PRE_SCREEN',n,'dates',len(q),'maxabs',round(m,6),'mean',round(np.mean(q),6))
print('MAX_ABS_LIBRARY_CORRELATION_PRE_SCREEN',round(mx,6))
