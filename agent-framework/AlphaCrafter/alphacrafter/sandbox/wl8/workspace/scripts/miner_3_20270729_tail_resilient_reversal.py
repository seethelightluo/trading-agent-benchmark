import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-28'); a=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 r=x.close.pct_change(); q=r.rolling(20,min_periods=15).quantile(.1).shift(1); vol=r.shift(1).rolling(20,min_periods=15).std()
 # fade only unusually negative recent 3d moves, normalized by lagged volatility
 r3=x.close.pct_change(3); x['sig']=(-r3/vol).where(r3.shift(1)<q*3,0.0)
 x['f1']=x.close.shift(-1)/x.close-1; x['f5']=x.close.shift(-5)/x.close-1; x['symbol']=s; a.append(x[['date','symbol','sig','f1','f5']])
z=pd.concat(a)
def calc(df,h):
 out=[];ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1:out.append(spearmanr(g.sig,g[h]).statistic);ns.append(len(g))
 v=np.array(out); return {'dates':len(v),'rows':len(df),'avg_n':round(np.mean(ns),2) if ns else 0,'ic':round(v.mean(),6) if len(v) else None,'icir':round(v.mean()/v.std(ddof=1),6) if len(v)>1 else None,'hit':round(np.mean(v>0),4) if len(v) else None}
print('overall1',calc(z,'f1'),'coverage',round(z.sig.notna().mean(),4)); print('overall5',calc(z,'f5'))
for q,c in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027)]:print(q,calc(z[c],'f1'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_3_20270729_tail_resilient_reversal_signal.csv',index=False)
