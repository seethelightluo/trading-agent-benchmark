import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2035-10-11'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change()
# Candidate: cross-asset breadth-confirmed trend. A 20d return is scaled by
# contemporaneous breadth of positive 20d returns, rewarding trends supported broadly.
ret20=P.pct_change(20).shift(1); breadth=(ret20>0).sum(axis=1)/ret20.notna().sum(axis=1)
F=ret20.mul((0.5+ breadth),axis=0)
print('dates',len(P),'instruments',len(D),'cutoff',P.index.max())
for h in [1,3,5,10,20]:
 vals=[]; cov=[]; dates=[]
 for i in range(65,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);cov.append(len(z)/len(U));dates.append(P.index[i])
 a=pd.Series(vals); recent=a.iloc[-252:]
 print('h',h,'n',len(a),'all_ic',round(a.mean(),6),'all_icir',round(a.mean()/a.std(),4),'recent_ic',round(recent.mean(),6),'recent_icir',round(recent.mean()/recent.std(),4),'hit',round((recent>0).mean(),4),'cov',round(np.mean(cov),4))
# rank turnover: fraction whose cross-sectional rank changes materially day to day
r=F.rank(axis=1,pct=True); turn=(r.diff().abs().mean(axis=1)>0.10).mean()
print('turnover_proxy',round(turn,5))
F.index.name='date'; F.to_csv('scripts/miner_1_20351012_breadth_confirmed_trend_signal.csv')
