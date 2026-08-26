import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2033-10-30']; r=px.pct_change()
# A simple interpretable trend-quality factor: medium momentum, penalized by downside risk,
# and gated by cross-asset breadth. The gate is lagged one session.
ret20=px.pct_change(20); down=r.clip(upper=0).rolling(40,min_periods=25).std()*np.sqrt(252)
trend_quality=ret20/(down+1e-12)
breadth=(r.rolling(20,min_periods=15).mean()>0).mean(axis=1)
gate=(breadth>=breadth.rolling(120,min_periods=60).median()).shift(1)
sig=trend_quality.mul(gate,axis=0).shift(1)
# signal itself uses only t-1 and forward returns start t

def ev(x,h):
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z); ns.append(len(q))
 a=np.array(vals)
 return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/(a.std(ddof=1)+1e-12)),float((a>0).mean())
print('candidate breadth-gated downside-quality momentum; assets',len(A),'dates',len(px),'cutoff',px.index[-1].date())
print('coverage',float(sig.notna().mean().mean()),'gate-active',float(gate.mean()),'turnover',float(sig.rank(axis=1,pct=True).diff().abs().stack().mean()))
for h in [1,5,10,20]: print('h',h,ev(sig,h))
for n in [180,500,750]: print('recent',n,ev(sig.iloc[-n:],10))
sig.to_csv('scripts/miner_1_20331031_breadth_downside_quality_signal.csv')
