import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-06-16')
a=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date'); x['r']=x.close.pct_change(); x['tr']=np.maximum(x.high-x.low,np.maximum(abs(x.high-x.close.shift()),abs(x.low-x.close.shift()))); x['atr']=x.tr.shift(1).rolling(20,min_periods=15).mean(); x['sig']=-(x.open-x.close.shift(1))/x.atr; x['fwd']=x.close.shift(-1)/x.close-1; x['fwd5']=x.close.shift(-5)/x.close-1; x['symbol']=s; a.append(x[['date','symbol','sig','fwd','fwd5']])
z=pd.concat(a)
def calc(q,y):
 vals=[]; ns=[]
 for d,g in q.dropna(subset=['sig',y]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[y].nunique()>1: vals.append(spearmanr(g.sig,g[y]).statistic);ns.append(len(g))
 v=np.array(vals); return len(v),round(np.mean(ns),2),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6),round(np.mean(v>0),4)
print('rows',len(z),'dates',z.date.nunique(),'coverage',z.sig.notna().mean()); print('daily',calc(z,'fwd')); print('5d',calc(z,'fwd5'))
for lab,q in [('2020-22',z[z.date.dt.year<=2022]),('2023-25',z[z.date.dt.year.between(2023,2025)]),('2026',z[z.date.dt.year==2026]),('2027',z[z.date.dt.year==2027])]: print(lab,calc(q,'fwd'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_1_20270617_atr_scaled_intraday_reversal_signal.csv',index=False)
