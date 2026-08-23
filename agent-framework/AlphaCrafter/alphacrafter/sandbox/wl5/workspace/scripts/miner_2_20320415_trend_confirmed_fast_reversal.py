import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4200); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
# Candidate: fast reversal scaled by medium-horizon volatility, with a slow-trend
# confirmation that avoids buying persistent losers: 5d reversal, multiplied by sign(60d trend).
for i in range(95,len(P)-11):
 rev=-R.iloc[i-5:i].sum()
 vol=R.iloc[i-60:i].std().replace(0,np.nan)
 slow=np.sign(R.iloc[i-60:i].sum())
 f=(rev/(vol+1e-12)*slow).replace([np.inf,-np.inf],np.nan).rank(pct=True)
 for s,v in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v)})
 y=P.iloc[i+10]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); print('universe',15,'usable_dates',len(x),'data_end',P.index.max().date())
m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('horizon',10,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-04-14')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None,'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6) if len(w)>1 else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_2_20320415_trend_confirmed_fast_reversal_signal.csv',index=False)
