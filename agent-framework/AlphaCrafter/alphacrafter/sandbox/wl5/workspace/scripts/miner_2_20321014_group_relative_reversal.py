import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; G={s:('eq' if s in U[:8] else 'com' if s in U[8:11] else 'crypto' if s in U[11:13] else 'yield') for s in U}
C={}
for s in U:
 d=get_stock_daily_data(s,days=4500); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
for i in range(90,len(P)-11):
 r=R.iloc[:i]; gr={g:r.iloc[-20:][[s for s in U if G[s]==g]].mean(axis=1).sum() for g in set(G.values())}
 # contrarian signal to own group-vs-other group relative move; mild group dispersion gate
 vals=np.array(list(gr.values())); disp=np.std(vals)
 f=pd.Series({s:-(gr[G[s]]-np.mean([gr[g] for g in gr if g!=G[s]]))*(1+min(disp/0.03,1)) for s in U}).rank(pct=True)
 for s,val in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(val)})
 y=P.iloc[i+10]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'data_end',P.index.max().date()); print('horizon',10,'IC',round(m,6),'ICIR',round(m/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),5));
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-10-13')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None,'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6) if len(w)>1 else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6)); pd.DataFrame(sig).to_csv('scripts/miner_2_20321014_group_relative_reversal_signal.csv',index=False)
