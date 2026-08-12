import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100:d=get_index_daily_data(s,1500)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float).reindex(P.index).ffill()
rows=[]; sigs=[]
for t in range(85,len(P)-1):
 if pd.isna(v.iloc[t]):continue
 # stress intensity based only on completed VIX observations, clipped and smoothed
 vm=v.iloc[max(0,t-59):t+1].median(); stress=2.0 if v.iloc[t]>vm else 0.0
 m=R.mean(axis=1); vals={}
 for s in P:
  z=pd.concat([R[s].iloc[t-59:t+1],m.iloc[t-59:t+1]],axis=1).dropna()
  if len(z)<30 or z.iloc[:,1].var()<=1e-12:continue
  b=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var(); vol=z.iloc[:,0].std()
  res=(R[s].iloc[t-4:t+1]-b*m.iloc[t-4:t+1]).sum()
  if vol>1e-8: vals[s]=-res/vol*stress
 q=pd.concat([pd.Series(vals),R.iloc[t+1].reindex(vals)],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].std()>1e-12 and q.iloc[:,1].std()>1e-12:
  ic=q.iloc[:,0].corr(q.iloc[:,1])
  if np.isfinite(ic): rows.append((P.index[t],ic,len(q)));sigs.append(pd.Series(vals,name=P.index[t]))
a=np.array([x[1] for x in rows]);n=np.array([x[2] for x in rows])
print('dates',len(a),'assets',len(P.columns),'avgN',n.mean(),'coverage',n.mean()/len(P.columns))
print('IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for lab,cut in [('2027+',pd.Timestamp('2027-01-01')),('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
 b=a[[x[0]>=cut for x in rows]];print(lab,len(b),'IC %.6f ICIR %.6f'%(b.mean(),b.mean()/b.std(ddof=1)))
S=pd.DataFrame(sigs);print('turnover',S.rank(pct=True).diff().abs().mean().mean())
for h in [3,5,10]:
 z=[]
 for date,_,_, in rows:
  j=P.index.get_loc(date)
  if j+h>=len(P):continue
  f=S.loc[date]; fw=(P.iloc[j+h]/P.iloc[j]-1); q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].std()>1e-12 and q.iloc[:,1].std()>1e-12:
   ic=q.iloc[:,0].corr(q.iloc[:,1])
   if np.isfinite(ic): z.append(ic)
 print('decay',h,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1))
pd.DataFrame(sigs).to_csv('scripts/miner_1_20290906_vix_binary_residual_signal.csv',index_label='date')
