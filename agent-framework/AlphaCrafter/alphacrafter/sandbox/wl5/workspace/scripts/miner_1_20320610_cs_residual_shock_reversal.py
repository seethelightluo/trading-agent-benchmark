import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
# Cross-sectional residual shock: remove contemporaneous common median return, reverse only
# sizeable 20d idiosyncratic shocks; rank is interpretable and low turnover.
for i in range(65,len(P)-11):
 hist=R.iloc[i-60:i]
 csmed=hist.median(axis=1)
 resid=hist.sub(csmed,axis=0)
 shock=resid.iloc[-20:].sum()
 scale=resid.iloc[-60:].std().replace(0,np.nan)*np.sqrt(20)
 f=(-shock/scale).replace([np.inf,-np.inf],np.nan).rank(pct=True)
 for s,v in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v)})
 y=(P.iloc[i+10]/P.iloc[i]-1).reindex(f.index); z=pd.concat([f.rename('f'),y.rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'data_end',P.index.max().date())
print('horizon',10,'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-06-01')]:
 w=x[(x.date>=a)&(x.date<=b)];print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6),'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6) if len(w)>1 else np.nan)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal');print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_1_20320610_cs_residual_shock_reversal_signal.csv',index=False)
