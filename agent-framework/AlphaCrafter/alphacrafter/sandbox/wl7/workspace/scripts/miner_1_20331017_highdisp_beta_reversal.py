import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2033-10-16']; r=px.pct_change(); bench=r.mean(axis=1)
cov=r.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
br=(1+bench).rolling(10).apply(np.prod,raw=True)-1
resid=px.pct_change(10)-beta.mul(br,axis=0)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(252)
base=-resid/(vol+1e-12)
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
# cross-sectional high-dispersion gate, lagged to ensure completed-day information
threshold=disp.rolling(120,min_periods=60).median()
sig=base.where(disp>threshold,0.0).shift(1)
f=px.shift(-10)/px-1

def eval(x):
 vals=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z);ns.append(len(q))
 a=np.array(vals)
 return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean(),[float(y.mean()) for y in np.array_split(a,3)]
print('candidate high-dispersion residual reversal; assets',len(A),'dates',len(px),'cutoff',px.index[-1].date())
print('coverage',round(sig.notna().mean().mean(),4),'active coverage',(sig!=0).mean().mean(),'avg active N',round((sig!=0).sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; z=eval(sig); print('h',h,z)
# restore h10 and recent diagnostics
f=px.shift(-10)/px-1
for n in [180,500,750]:
 z=eval(sig.iloc[-n:]); print('recent',n,z)
print('turnover',float(sig.rank(axis=1,pct=True).diff().abs().stack().mean()))
sig.to_csv('scripts/miner_1_20331017_highdisp_beta_reversal_signal.csv')
