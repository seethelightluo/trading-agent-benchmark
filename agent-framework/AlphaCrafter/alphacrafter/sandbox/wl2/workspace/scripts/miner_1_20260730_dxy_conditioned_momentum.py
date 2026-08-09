import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(p):
 d=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close; return d.loc[:cut]
P=pd.concat({a:load('../persistent/stock_data/'+a+'.csv') for a in U},axis=1).sort_index(); R=P.pct_change(); D=load('../persistent/index_data/DXY.csv').reindex(P.index).ffill().pct_change()
reg=-D.rolling(20,min_periods=15).sum(); F=R.rolling(20,min_periods=15).sum().mul(reg,axis=0); Y=R.shift(-1)
ics=[];ds=[];ns=[]
for dt in F.index.intersection(Y.index):
 z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
 if len(z)>=8:ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
a=np.array(ics);print('dates',len(a),'range',min(ds),max(ds),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()));print('coverage',F.stack().notna().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for y in range(2020,2027):
 v=a[[d.year==y for d in ds]];print(y,len(v),round(v.mean(),5) if len(v) else None,round(v.mean()/v.std(ddof=1),4) if len(v)>1 else None)
