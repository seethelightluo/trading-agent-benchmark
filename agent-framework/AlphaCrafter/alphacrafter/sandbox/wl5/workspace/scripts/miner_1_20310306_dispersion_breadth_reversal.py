import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); res=R.sub(R.median(axis=1),axis=0)
# Novel idea: reversal amplified when the cross-section is unusually dispersed,
# but attenuated during broad directional breadth. All state variables use prior closes.
def factor(i):
 r5=res.iloc[i-4:i+1].sum(); v60=res.iloc[i-59:i+1].std(); base=-r5/(v60+1e-12)
 disp5=res.iloc[i-4:i+1].std(axis=1).mean(); disp120=res.iloc[i-119:i+1].std(axis=1).mean()
 dispersion=np.clip(disp5/(disp120+1e-12),0.5,2.0)
 breadth=(R.iloc[i-19:i+1].median(axis=1)>0).mean()
 # reversal works best away from one-sided breadth; continuous center weighting
 breadth_gate=np.clip(1.35-0.7*abs(breadth-0.5),0.9,1.35)
 return base*dispersion*breadth_gate

def run(h):
 rows=[]; sig=[]
 for i in range(120,len(P)-h):
  f=factor(i); y=P.iloc[i+h]/P.iloc[i]-1
  z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
  if h==5:
   sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
 x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
 print('horizon',h,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-03-05')]:
  q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6) if len(q) else None,'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else None)
 if h==5:
  S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6)); pd.DataFrame(sig).to_csv('scripts/miner_1_20310306_dispersion_breadth_reversal_5d_signal.csv',index=False)
for h in [5,10,20]: run(h)
