import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date)
 C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
# Candidate: asymmetric shock reversal. Fade recent residual shocks, but normalize by
# downside semideviation (causal), making large downside-risk assets less dominant.
def fac(i):
 r5=R.iloc[i-4:i+1].sum(); resid=r5-r5.median()
 dn=R.iloc[i-59:i+1].where(R.iloc[i-59:i+1]<0).std()+1e-12
 vol=R.iloc[i-59:i+1].std()+1e-12
 # blend downside-risk normalization with total risk to avoid unstable tails
 scale=0.7*dn+0.3*vol
 return -resid/scale

def run(h):
 rows=[]; sig=[]
 for i in range(80,len(P)-h):
  f=fac(i); y=P.iloc[i+h]/P.iloc[i]-1
  z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
  if h==5: sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
 x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
 print('horizon',h,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-04-16')]:
  q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6) if len(q) else None)
 if h==5:
  S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal')
  print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
  pd.DataFrame(sig).to_csv('scripts/miner_1_20310417_asymshock_reversal_5d_signal.csv',index=False)
for h in [5,10,20]: run(h)
