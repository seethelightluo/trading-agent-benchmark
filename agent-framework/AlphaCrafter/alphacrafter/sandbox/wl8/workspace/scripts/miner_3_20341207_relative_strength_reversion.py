import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Cross-asset relative-strength reversal: fade an extreme 60d winner/loser only
# when the market breadth (share above 20d mean) is weak/strong respectively.
ret60=p.pct_change(60); med=ret60.median(axis=1); rel=ret60.sub(med,axis=0)
breadth=(p>p.rolling(20,min_periods=15).mean()).mean(axis=1)
# Reversal is strongest at cross-sectional extremes and is damped in ambiguous breadth.
sig=(-rel * (0.5+abs(breadth-0.5))).shift(1)
rows=[]; cov=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],(p.shift(-10)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z))); cov.append(sig.loc[dt].notna().mean())
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':].dropna(); v=ic.ic
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',np.mean(cov),'IC',v.mean(),'dailyICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
ranks=sig.rank(axis=1,pct=True); tt=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8: tt.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('turnover',np.mean(tt))
for w in [365,750,1260]:
 q=v.tail(w); print('recent',w,'ICIR',q.mean()/q.std(ddof=1))
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(rr))
out=sig.loc[ic.index].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20341207_relative_strength_reversion_signal.csv',index=False)
ic.reset_index().to_csv('scripts/miner_3_20341207_relative_strength_reversion_ic.csv',index=False)
