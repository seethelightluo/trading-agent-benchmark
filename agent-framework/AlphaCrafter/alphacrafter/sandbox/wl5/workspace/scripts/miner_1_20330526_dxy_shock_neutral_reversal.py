import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:C[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); d=pd.read_csv('../persistent/index_data/DXY.csv'); d.date=pd.to_datetime(d.date); dx=d.set_index('date').close.astype(float).reindex(P.index).ffill(); dr=np.log(dx).diff(); rows=[]; sig=[]
for i in range(90,len(P)-11):
 rr=R.iloc[i-59:i+1]; dd=dr.iloc[i-59:i+1]; b=rr.apply(lambda x:x.cov(dd)/(dd.var()+1e-12)); move=R.iloc[i-9:i+1].sum()-b*dr.iloc[i-9:i+1].sum(); vol=rr.std()+1e-12; shock=abs(dd.iloc[-10:].sum())/(dd.std()*np.sqrt(10)+1e-12); f=(-move/(vol*np.sqrt(10)))*float(np.clip(shock,0.5,2.0)); f=f.replace([np.inf,-np.inf],np.nan); y=R.iloc[i+1:i+11].sum(axis=0); z=pd.concat([f.rename('f'),y.rename('y')],axis=1).dropna();
 if len(z)>=8 and z.f.nunique()>1:rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items() if np.isfinite(a)]
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); a=x.ic.to_numpy(); m=a.mean(); print({'universe':15,'dates':len(x),'start':str(x.date.min().date()),'end':str(x.date.max().date()),'mean_n':x.n.mean(),'coverage':x.n.mean()/15,'IC':m,'ICIR':m/a.std(ddof=1)*np.sqrt(252),'hit':(a>0).mean()});
for aa,bb in [('2026-08-26','2029-12-31'),('2030-01-01','2031-12-31'),('2032-01-01','2033-05-11')]:
 q=x[(x.date>=aa)&(x.date<=bb)].ic;print('regime',aa,bb,len(q),q.mean() if len(q) else None,q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal');print('turnover',S.rank(axis=1,pct=True).diff().abs().mean().mean());pd.DataFrame(sig).to_csv('scripts/miner_1_20330526_dxy_shock_neutral_reversal_signal.csv',index=False)
