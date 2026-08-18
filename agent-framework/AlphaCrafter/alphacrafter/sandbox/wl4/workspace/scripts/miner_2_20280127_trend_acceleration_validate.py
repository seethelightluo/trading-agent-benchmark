import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); D[s]=x.close
P=pd.DataFrame(D).sort_index(); P=P.loc[:'2028-01-26']
# interpretable acceleration: recent 20d return minus slower 60d return
F=(P/P.shift(20)-1)-(P/P.shift(60)-1)
H=10; obs=[]; ns=[]; dates=[]
for i in range(60,len(P)-H):
 z=pd.concat([F.iloc[i],P.iloc[i+H]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  obs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(P.index[i])
a=np.asarray(obs); print('period',P.index.min().date(),P.index.max().date(),'dates',len(a),'avgN',round(np.mean(ns),2),'universe',len(U))
print('horizon',H,'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(pct=True).diff().abs().mean().mean(),4))
for name,mask in [('early',np.array(dates,dtype='datetime64[ns]')<np.datetime64('2022-01-01')),('2022-23',(np.array(dates,dtype='datetime64[ns]')>=np.datetime64('2022-01-01'))&(np.array(dates,dtype='datetime64[ns]')<np.datetime64('2024-01-01'))),('2024-26',np.array(dates,dtype='datetime64[ns]')>=np.datetime64('2024-01-01'))]:
 q=a[mask]; print(name,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),4) if len(q)>1 else None)
print('recent250',round(a[-250:].mean(),6),'recent250_icir',round(a[-250:].mean()/a[-250:].std(ddof=1),4))
