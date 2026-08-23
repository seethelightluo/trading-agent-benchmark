import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
def factor(i):
 trend=np.log(P.iloc[i]/P.iloc[i-39]); vol=R.iloc[i-19:i+1].std()*np.sqrt(20); confirm=np.log(P.iloc[i]/P.iloc[i-9])<0
 return -trend/(vol+1e-12)*(0.5+0.5*confirm.astype(float))
for h in [5,10,20]:
 rows=[]; sig=[]
 for i in range(80,len(P)-h):
  f=factor(i); y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
  for s,a in f.items(): sig.append((P.index[i],s,float(a)))
 x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); ir=m/x.ic.std(ddof=1)
 print('horizon',h,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(ir,6),'hit',round((x.ic>0).mean(),5))
 if h==5: pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20310123_inverse_trend_40d_5d_signal.csv',index=False)
 if h==20: pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20310123_inverse_trend_40d_20d_signal.csv',index=False)
