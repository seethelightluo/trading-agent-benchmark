import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
data={}
for s in U:
 p=f'{base}/{s}.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
  data[s]=d['close'].astype(float)
px=pd.concat(data,axis=1).sort_index()
r=px.pct_change()
# risk-adjusted medium momentum, known interpretable candidate
fac=r.rolling(20,min_periods=15).sum()/r.rolling(20,min_periods=15).std()
# signal at t, forward return t+1 (data cutoff alignment)
ics=[]; rows=[]
for i in range(20,len(px)-1):
 f=fac.iloc[i]; y=r.iloc[i+1]
 z=pd.concat([f,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  ics.append(ic); rows.append((px.index[i],ic,len(z)))
a=np.array(ics); print('dates',len(a),'avg_names',np.mean([x[2] for x in rows]),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-12-31')]:
 q=[x[1] for x in rows if str(x[0])[:10]>=lo and str(x[0])[:10]<=hi]
 if q: print(lo,hi,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
for h in [5,10,20]:
 ys=px.shift(-h)/px-1; vals=[]
 for i in range(20,len(px)-h):
  z=pd.concat([fac.iloc[i],ys.iloc[i+h-1]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,len(vals),np.mean(vals),np.mean(vals)/np.std(vals,ddof=1))
print('coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
# artifact for provenance
out=[]
for i in range(len(px)):
 for s in U:
  if pd.notna(fac.iloc[i].get(s)): out.append({'date':px.index[i].date(),'symbol':s,'value':fac.iloc[i][s]})
pd.DataFrame(out).to_csv('../persistent/factor_signals_miner_2_20270225_ra_mom20.csv',index=False)
