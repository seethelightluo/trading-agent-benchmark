import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(p):
 d=pd.read_csv(p,parse_dates=['date']);d['date']=pd.to_datetime(d.date).dt.normalize();return d.set_index('date').sort_index()
D={s:L('../persistent/stock_data/'+s+'.csv') for s in U};v=L('../persistent/index_data/VIX.csv').close;end=pd.Timestamp('2027-06-16'); dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U});R=C.pct_change();vx=v.reindex(dates).ffill();g=(vx>vx.rolling(60,min_periods=30).median()).astype(float)
F=((-R.rolling(3,min_periods=3).sum()).mul(g,axis=0)+R.rolling(20,min_periods=15).sum().mul(1-g,axis=0)).shift(1);Y=C.shift(-1).div(C)-1
a=[];ns=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
a=np.array(a);print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
