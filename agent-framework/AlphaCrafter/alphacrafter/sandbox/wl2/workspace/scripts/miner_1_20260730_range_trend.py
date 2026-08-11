import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-07-15')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'];return d.loc[:cut]
P={a:load(a) for a in U}; F=pd.DataFrame({a:(p-p.rolling(60,min_periods=45).min())/(p.rolling(60,min_periods=45).max()-p.rolling(60,min_periods=45).min()) for a,p in P.items()});R=pd.DataFrame({a:p.pct_change(fill_method=None) for a,p in P.items()})
def test(Y):
 vals=[];ds=[];ns=[]
 for dt in F.index.intersection(Y.index):
  z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 v=np.array(vals);return v,ds,ns
v,ds,ns=test(R.shift(-1));print('factor range_trend_60d dates',len(v),'range',min(ds),max(ds),'avgN',np.mean(ns));print('IC %.6f ICIR %.6f hit %.4f'%(v.mean(),v.mean()/v.std(ddof=1),(v>0).mean()));print('coverage %.4f turnover %.4f'%(F.stack().notna().mean(),F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for y in range(2020,2027):
 x=v[[d.year==y for d in ds]];print(y,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
for h in [5,10]:
 x,_,n=test(R.shift(-h));print('horizon',h,'dates',len(x),'avgN',np.mean(n),'IC %.6f ICIR %.6f'%(x.mean(),x.mean()/x.std(ddof=1)))
F.reset_index().rename(columns={'index':'date'}).to_csv('/tmp/range_trend_60d_signal.csv',index=False);print('signal_artifact /tmp/range_trend_60d_signal.csv')
