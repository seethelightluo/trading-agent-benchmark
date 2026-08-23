import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-28')
parts=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 x['ret']=x.close.pct_change(); x['mom']=x.close/x.close.shift(5)-1
 # volume surprise is lagged: completed-session volume relative to its prior 60d median
 med=x.volume.shift(1).rolling(60,min_periods=30).median(); x['vs']= (x.volume/med).clip(0.25,4.0).shift(1)
 x['sig']=x.mom.shift(1)*np.log(x.vs)
 x['f1']=x.close.shift(-1)/x.close-1; x['f5']=x.close.shift(-5)/x.close-1
 x['symbol']=s; parts.append(x[['date','symbol','sig','f1','f5']])
z=pd.concat(parts)
def calc(df,h):
 vals=[]; ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1: vals.append(spearmanr(g.sig,g[h]).statistic); ns.append(len(g))
 v=np.asarray(vals)
 return {'dates':len(v),'rows':len(df),'avg_n':round(float(np.mean(ns)),2),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float(np.mean(v>0)),4)}
print('overall_1d',calc(z,'f1'),'coverage',round(float(z.sig.notna().mean()),4))
print('overall_5d',calc(z,'f5'))
for q,c in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027)]: print(q,calc(z[c],'f1'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_3_20270729_volume_confirmed_momentum_signal.csv',index=False)
