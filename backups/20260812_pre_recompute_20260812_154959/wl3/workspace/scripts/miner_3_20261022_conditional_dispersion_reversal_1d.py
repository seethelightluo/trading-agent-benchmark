import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-10-21'; rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); r=x.close.pct_change(); rv=r.rolling(20,min_periods=12).std()
 rows.append(pd.DataFrame({'date':x.date,'symbol':s,'f':-r/(rv+1e-12),'r':r,'y':x.close.shift(-1)/x.close-1}))
a=pd.concat(rows,ignore_index=True); disp=a.groupby('date')['r'].apply(lambda z:z.abs().mean()).sort_index(); hist=disp.rolling(60,min_periods=30).median().shift(1); active=(disp>hist).rename('active'); a=a.join(active,on='date').dropna(subset=['f','y','active']); A=a[a.active].copy()
def calc(df,col='f',horizon='y'):
 vals=[]; ns=[]
 for d,g in df.groupby('date'):
  if len(g)>=8 and g[col].nunique()>1 and g[horizon].nunique()>1: vals.append((d,spearmanr(g[col],g[horizon]).statistic)); ns.append(len(g))
 z=pd.DataFrame(vals,columns=['date','ic']).set_index('date'); q=z.ic
 return z,ns,q
z,ns,q=calc(A); rank=A.assign(rank=A.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank')
print('candidate conditional_dispersion_reversal_1d cutoff',cut,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(len(A)/sum(len(pd.read_csv('../persistent/stock_data/'+s+'.csv')) for s in U),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),6))
for name,lo,hi in [('early','2020','2022-12-31'),('mid','2023','2024-12-31'),('late','2025','2026-10-21')]:
 v=z.loc[lo:hi].ic; print('regime',name,len(v),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6))
for h in [3,5,10]:
 rr=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); r=x.close.pct_change(); rv=r.rolling(20,min_periods=12).std(); rr.append(pd.DataFrame({'date':x.date,'f':-r/(rv+1e-12),'r':r,'y':x.close.shift(-h)/x.close-1}))
 aa=pd.concat(rr,ignore_index=True); aa=aa.join(active,on='date').dropna(); aa=aa[aa.active]; vv=[]
 for d,g in aa.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vv.append(spearmanr(g.f,g.y).statistic)
 vv=np.array(vv); print('decay',h,len(vv),round(vv.mean(),6),round(vv.mean()/vv.std(ddof=1),6))
A[['date','symbol','f']].rename(columns={'f':'signal'}).to_csv('scripts/miner_3_20261022_conditional_dispersion_reversal_1d_signal.csv',index=False)
