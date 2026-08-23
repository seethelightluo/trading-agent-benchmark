import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
def factor(i):
 vol=R.iloc[i-59:i+1].std()*np.sqrt(60)+1e-12
 x=R.iloc[i-4:i+1].sum(); base=-(x-x.median())/vol
 disp=R.iloc[i-19:i+1].std(axis=1).mean(); hist=R.iloc[i-59:i+1].std(axis=1).rolling(20).mean().dropna()
 # amplify reversal in elevated cross-asset dispersion, retain half strength otherwise
 gate=1.0 if disp>hist.median() else 0.5
 return base*gate
rows=[]; sig=[]
for i in range(80,len(P)-5):
 f=factor(i); y=P.iloc[i+5]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('dates',len(x),'meanN',x.n.mean(),'coverage',x.n.sum()/(len(x)*15),'IC',m,'ICIR',m/sd,'hit',(x.ic>0).mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-03-19')]:
 q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',q.ic.mean())
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
pd.DataFrame(sig).to_csv('scripts/miner_1_20310403_highdisp_residual_reversal_5d_signal.csv',index=False)
