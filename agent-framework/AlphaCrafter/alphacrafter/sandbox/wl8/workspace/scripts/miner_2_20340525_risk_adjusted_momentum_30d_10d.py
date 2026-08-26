import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-05-24'); start=pd.Timestamp('2026-07-16'); base='../persistent/stock_data'
px={}
for s in SYMS:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close[d.index<=cut]
prices=pd.DataFrame(px); ret=prices.pct_change(); mom=prices/prices.shift(30)-1; rv=ret.rolling(30,min_periods=20).std(); f=(mom/rv).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(prices)-10):
 if prices.index[i]<start: continue
 z=pd.concat([f.iloc[i],prices.iloc[i+10]/prices.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((prices.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); rank=f.rank(axis=1,pct=True); to=rank.diff().abs().mean(axis=1).loc[start:].dropna().mean()
print('dates',len(r),'period',r.index.min().date(),r.index.max().date(),'avgN',round(r.n.mean(),2),'coverage',round(r.n.sum()/(len(r)*15),4))
print('IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4),'turnover',round(to,6))
for h in [1,5,20]:
 rr=[]
 for i in range(len(prices)-h):
  if prices.index[i]<start: continue
  z=pd.concat([f.iloc[i],prices.iloc[i+h]/prices.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,round(float(np.nanmean(rr)),6),'n',len(rr))
print('regimes',[(str(a.date()),str(b.date()),round(r.loc[(r.index>=a)&(r.index<=b),'ic'].mean(),6),len(r.loc[(r.index>=a)&(r.index<=b)])) for a,b in [(pd.Timestamp('2026-07-16'),pd.Timestamp('2028-12-31')),(pd.Timestamp('2029-01-01'),pd.Timestamp('2031-12-31')),(pd.Timestamp('2032-01-01'),cut)]])
out='scripts/miner_2_20340525_risk_adjusted_momentum_30d_10d'; r.to_csv(out+'_ic.csv'); f.to_csv(out+'_signal.csv'); print('artifacts',out+'_ic.csv',out+'_signal.csv')
