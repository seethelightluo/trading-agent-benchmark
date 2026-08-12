import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(p); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index().ffill(); ret=px.pct_change()
# downside-risk-adjusted intermediate momentum, lagged one completed day
r30=px.pct_change(30); down=ret.where(ret<0).rolling(40, min_periods=10).std(); f=(r30/down).shift(1)
rows=[]; turnovers=[]; prev=None
for i in range(45,len(px)-10):
 vals=f.iloc[i]; fw=px.iloc[i+10]/px.iloc[i]-1
 z=pd.concat([vals,fw],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((px.index[i],ic,len(z)))
 # turnover of cross-sectional ranks
 ranks=vals.rank(pct=True)
 if prev is not None: turnovers.append(np.nanmean(abs(ranks-prev)))
 prev=ranks
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
for label,sub in [('all',q),('2020-25',q.loc[:'2025-12-31']),('2026+',q.loc['2026-01-01':]),('2029+',q.loc['2029-01-01':]),('2030+',q.loc['2030-01-01':])]:
 m=sub.ic.mean(); sd=sub.ic.std(ddof=1); print(label,'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252/10),6),'hit',round((sub.ic>0).mean(),4))
print('turnover',np.nanmean(turnovers))
# decay diagnostics
for h in [1,5,10,20]:
 rr=[]
 for i in range(45,len(px)-h):
  z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'ic',np.nanmean(rr),'n',len(rr))
