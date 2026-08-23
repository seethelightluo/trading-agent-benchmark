import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
# Revalidation of effective half-amplitude downside-skew recovery reversal.
for i in range(25,len(P)-21):
 r3=R.iloc[i-2:i+1].sum(); v=R.iloc[i-20:i].std()+1e-12; dn=(R.iloc[i-20:i]<0).sum(); up=(R.iloc[i-20:i]>0).sum(); amp=1+.50*((dn-up)/20); f=(-r3/v*amp).replace([np.inf,-np.inf],np.nan)
 for h in (10,):
  y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1); print('universe',15,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5),'data_end',P.index.max().date())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-01-07')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6)); pd.DataFrame(sig).to_csv('scripts/miner_3_20320108_downside_skew_halfamp_reval_signal.csv',index=False)
