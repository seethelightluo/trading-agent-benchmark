import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4600); d.date=pd.to_datetime(d.date); C[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
for i in range(130,len(P)-11):
 rr=R.iloc[:i]; market=rr.iloc[-80:].mean(axis=1); out=[]
 for s in U:
  z=rr[s].iloc[-80:]; mm=market.loc[z.index]; beta=z.cov(mm)/(mm.var()+1e-12); resid=z-beta*mm
  f=-resid.iloc[-60:].sum()/(resid.iloc[-60:].std()+1e-12); out.append(f)
 f=pd.Series(out,index=U).rank(pct=True)
 for s,x in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(x)})
 y=P.iloc[i+10]/P.iloc[i]-1; q=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(q)>=8: rows.append((P.index[i],len(q),q.f.corr(q.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'horizon',10,'IC',round(m,6),'ICIR',round(m/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-01-19')]:
 w=x[(x.date>=a)&(x.date<=b)];print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6),'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6))
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal');print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6));pd.DataFrame(sig).to_csv('scripts/miner_1_20330120_residual60_signal.csv',index=False)
