import pandas as pd, numpy as np, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2033-10-30']; r=px.pct_change(); bench=r.mean(axis=1)
cov=r.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
br=(1+bench).rolling(10).apply(np.prod,raw=True)-1
resid=px.pct_change(10)-beta.mul(br,axis=0)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(252)
base=-resid/(vol+1e-12); disp=r.rolling(20,min_periods=15).std().mean(axis=1)
threshold=disp.rolling(120,min_periods=60).median(); sig=base.where(disp>threshold,0.0).shift(1)
def ev(x,h):
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z);ns.append(len(q))
 a=np.array(vals); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/(a.std(ddof=1)+1e-12)),float((a>0).mean())
print('revalidation factor high-dispersion beta-neutral reversal; assets',len(A),'dates',len(px),'cutoff',px.index[-1].date())
print('coverage',float(sig.notna().mean().mean()),'active',float((sig!=0).mean().mean()),'turnover',float(sig.rank(axis=1,pct=True).diff().abs().stack().mean()))
for h in [1,5,10,20]: print('h',h,ev(sig,h))
for n in [180,500,750]: print('recent',n,ev(sig.iloc[-n:],10))
sig.to_csv('scripts/miner_1_20331031_highdisp_beta_reversal_signal.csv')
