import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); res=R.sub(R.median(axis=1),axis=0)
# Candidate: residual short-term reversal with a causal market-trend stress gate.
# The gate increases reversal exposure after broad market drawdowns, while remaining
# continuous and bounded; this tests whether reversal is state dependent.
def factor(i):
 short=-res.iloc[i-4:i+1].sum(); vol20=res.iloc[i-19:i+1].std()
 base=short/(vol20+1e-12)
 m20=R.iloc[i-19:i+1].median(axis=1).sum()
 mvol=R.iloc[i-59:i+1].median(axis=1).std()
 stress=np.clip(-m20/(mvol*np.sqrt(20)+1e-12),-2,2)
 gate=np.clip(1+0.35*stress,0.3,1.7)
 return base*gate

def obs(h):
 rows=[]; sig=[]
 for i in range(120,len(P)-h):
  f=factor(i); y=P.iloc[i+h]/P.iloc[i]-1
  z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
  if h==5:
   for s,a in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(a)})
 x=pd.DataFrame(rows,columns=['date','n','ic']).dropna();m=x.ic.mean();sd=x.ic.std(ddof=1)
 print('horizon',h,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-02-19')]:
  q=x[(x.date>=a)&(x.date<=b)];print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6) if len(q) else None)
 if h==5:
  S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal');print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6));pd.DataFrame(sig).to_csv('scripts/miner_3_20310220_market_stress_gated_reversal_5d_signal.csv',index=False)
for h in [5,10,20]: obs(h)
