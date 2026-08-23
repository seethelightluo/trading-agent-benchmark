import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date)
 C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
def fac(i):
 # Relative 20-day reversal, normalized by recent downside risk and residualized to cross-sectional median.
 mom=P.iloc[i]/P.iloc[i-20]-1
 down=R.iloc[i-39:i+1].where(R.iloc[i-39:i+1]<0).std()*np.sqrt(40)
 f=-mom/(down+1e-12)
 return f-f.median()
def calc(h):
 rows=[]; sig=[]
 for i in range(61,len(P)-h):
  f=fac(i); y=P.iloc[i+h]/P.iloc[i]-1
  z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
  if h==10:
   for s,v in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v)})
 x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
 print('H',h,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-10-30')]:
  z=x[(x.date>=a)&(x.date<=b)]
  if len(z)>1: print('REG',a,b,'dates',len(z),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),5))
 if h==10:
  S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal')
  print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
  pd.DataFrame(sig).to_csv('scripts/miner_1_20301031_relative_downside_reversal_signal.csv',index=False)
for h in [5,10,20]: calc(h)
