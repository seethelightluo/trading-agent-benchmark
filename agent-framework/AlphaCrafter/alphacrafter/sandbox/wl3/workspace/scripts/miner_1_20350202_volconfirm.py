import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
syms=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; close={};vol={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  q=d.set_index(pd.to_datetime(d.date));close[s]=q.close.astype(float);vol[s]=q.volume.astype(float)
P=pd.DataFrame(close).sort_index(); V=pd.DataFrame(vol).reindex(P.index); r=P.pct_change()
# Volume-confirmed trend: 20-session return multiplied by signed volume surprise, lagged.
vz=(V/V.rolling(60,min_periods=30).mean()-1).clip(-2,2)
f=(r.rolling(20,min_periods=15).sum()*vz).shift(1)
ics=[];ns=[];tos=[]
for i in range(80,len(P)-10):
 n=f.columns[f.iloc[i].notna()&P.iloc[i].notna()&P.iloc[i+10].notna()]
 if len(n)<8:continue
 a=f.iloc[i][n];y=P.iloc[i+10][n]/P.iloc[i][n]-1;ics.append((P.index[i],a.corr(y,method='spearman')));ns.append(len(n))
 if i>80:tos.append(np.mean(abs(a.rank(pct=True)-f.iloc[i-1][n].rank(pct=True))))
ser=pd.Series(dict(ics)).dropna();print('UNIVERSE dates',len(P),'instruments',len(syms),'available',len(P.columns),'period',P.index[0],P.index[-1])
for l,z in [('all',ser),('recent120',ser.tail(120)),('recent252',ser.tail(252)),('recent504',ser.tail(504))]:print(l,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('avg_valid',round(np.mean(ns),3),'coverage',round(np.mean(ns)/len(syms),4),'turnover',round(np.mean(tos),4))
for j,z in enumerate(np.array_split(ser,4),1):print('block',j,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
pd.DataFrame({'date':ser.index.astype(str),'signal_ic':ser.values}).to_csv('scripts/miner_1_20350202_volconfirm_signal.csv',index=False)
