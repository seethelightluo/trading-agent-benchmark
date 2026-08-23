import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4500); d.date=pd.to_datetime(d.date)
 C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); M=R.mean(axis=1)
rows=[]; sig=[]
for i in range(70,len(P)-11):
 rr=R.iloc[i-60:i]; mm=M.iloc[i-60:i]
 beta=rr.apply(lambda x:x.cov(mm)/(mm.var()+1e-12))
 resid=rr.sub(mm,axis=0).mul(1-beta,axis=1)
 recent=resid.iloc[-5:].sum()
 # downside semideviation estimated only from prior negative residual observations
 down=resid.iloc[-40:].where(resid.iloc[-40:]<0).std().replace(0,np.nan)
 total=resid.iloc[-40:].std().replace(0,np.nan)
 # reversal is scaled by downside risk, with a mild asymmetry premium for downside-heavy assets
 asym=(down/(total+1e-12)).clip(0,2).fillna(1.0)
 f=(-recent/(total+1e-12))*asym
 f=f.rank(pct=True)
 for s,val in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(val)})
 y=P.iloc[i+10]/P.iloc[i]-1
 z=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'data_end',P.index.max().date())
print('horizon',10,'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-09-29')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None,'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6) if len(w)>1 else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_2_20320930_downside_residual_reversal_signal.csv',index=False)
