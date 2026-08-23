import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); M=R.mean(axis=1)
rows=[]; sig=[]
for i in range(80,len(P)-21):
 rr=R.iloc[i-59:i+1]; m=M.iloc[i-59:i+1]
 beta=rr.apply(lambda x:x.cov(m),axis=0)/(m.var()+1e-12)
 # Smooth the last five beta-adjusted shocks; reverse only the recent idiosyncratic move.
 resid=R.iloc[i-4:i+1]-pd.DataFrame({s:beta[s]*M.iloc[i-4:i+1] for s in U},index=R.index[i-4:i+1])
 vol=rr.std()+1e-12
 f=-resid.sum()/vol
 f=f.replace([np.inf,-np.inf],np.nan)
 y=P.iloc[i+5]/P.iloc[i]-1
 z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); print('data_end',P.index.max().date(),'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5))
q=x.ic; print('IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-06-11')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_1_20310612_smoothed_residual_reversal_5d_signal.csv',index=False)
