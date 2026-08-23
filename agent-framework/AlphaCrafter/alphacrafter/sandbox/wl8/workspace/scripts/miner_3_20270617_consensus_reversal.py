import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-06-16'); a=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date'); x['r3']=x.close/x.close.shift(3)-1; x['f1']=x.close.shift(-1)/x.close-1; x['f5']=x.close.shift(-5)/x.close-1; x['symbol']=s; a.append(x[['date','symbol','r3','f1','f5']])
z=pd.concat(a)
# lagged consensus: absolute mean signed return, normalized by trailing median; signal is reversal amplified by consensus.
cons=z.groupby('date').r3.apply(lambda q: abs(q.mean())).shift(1); med=cons.rolling(60,min_periods=30).median(); amp=(cons/med).clip(.5,2)
z=z.merge(amp.rename('amp'),left_on='date',right_index=True,how='left'); z['sig']=-z.r3*z.amp
def calc(df,h):
 v=[];ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1:v.append(spearmanr(g.sig,g[h]).statistic);ns.append(len(g))
 v=np.array(v);return len(v),len(df),round(np.mean(ns),2),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6),round((v>0).mean(),4)
print('f1',calc(z,'f1'),'coverage',round(z.sig.notna().mean(),4));print('f5',calc(z,'f5'))
for q,c in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027)]:print(q,calc(z[c],'f1'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_3_20270617_consensus_reversal_signal.csv',index=False)
