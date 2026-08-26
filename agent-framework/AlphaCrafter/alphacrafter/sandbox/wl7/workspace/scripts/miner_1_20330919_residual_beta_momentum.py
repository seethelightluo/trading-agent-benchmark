import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2033-09-18']; r=px.pct_change(); bench=r.mean(axis=1)
# residual momentum: asset 20d return stripped of rolling 60d beta to equal-weight cross-asset benchmark
cov=r.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
r20=px.pct_change(20); b20=(1+bench).rolling(20).apply(np.prod,raw=True)-1
vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
sig=(r20-beta.mul(b20,axis=0)).div(vol+1e-12).shift(1); sig.to_csv('scripts/miner_1_20330919_residual_beta_momentum_signal.csv')
print('assets',len(A),'dates',len(px),'cutoff',px.index[-1].date(),'coverage',round(sig.notna().mean().mean(),4),'avgN',round(sig.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z);ns.append(len(q))
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4),'thirds',[round(float(x.mean()),6) for x in np.array_split(a,3)])
for n in [180,500,750]:
 f=px.shift(-10)/px-1; vals=[]
 for dt in sig.index[-n:]:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z)
 a=np.array(vals); print('recent',n,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().stack().mean(),6))
# correlations to existing signal artifacts where overlap
for fn in ['miner_1_20330905_volscaled_acceleration_signal.csv','miner_1_20330822_momentum_acceleration_signal.csv']:
 try:
  old=pd.read_csv('scripts/'+fn,index_col=0,parse_dates=True); z=sig.stack().to_frame('new').join(old.stack().to_frame('old')).dropna().corr().iloc[0,1]; print('corr',fn,round(z,6))
 except Exception as e: print('corr',fn,'NA',e)
