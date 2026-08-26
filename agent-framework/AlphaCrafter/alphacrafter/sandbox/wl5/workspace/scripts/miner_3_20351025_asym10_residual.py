import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
def factor(i):
 r=R.iloc[i-10:i]; resid=r.sum()-r.sum().mean(); base=R.iloc[i-40:i]; dn=base.clip(upper=0).pow(2).mean().pow(.5); up=base.clip(lower=0).pow(2).mean().pow(.5); asym=(dn/(up+1e-8)).clip(0,8); vol=R.iloc[i-30:i].std()+1e-8
 return (-(resid/vol)*(1+0.8*(asym-1).clip(-1,3))).replace([np.inf,-np.inf],np.nan).clip(-10,10)
for i in range(80,len(P)-21):
 f=factor(i)
 for s,v in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v) if pd.notna(v) else np.nan})
 y=P.iloc[i+10]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); print('dates',len(x),'meanN',x.n.mean(),'coverage',x.n.mean()/15,'IC',m,'ICIR',m/x.ic.std(ddof=1),'hit',(x.ic>0).mean());
for a,b in [('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-10-01')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,len(w),w.ic.mean())
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()); pd.DataFrame(sig).to_csv('scripts/miner_3_20351025_asym10_residual_signal.csv',index=False)
