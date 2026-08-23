import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-14'); a=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date').copy(); x['sig']=-(x.close/x.close.shift(1)-1); x['fwd']=x.close.shift(-1)/x.close-1; x['fwd5']=x.close.shift(-5)/x.close-1; x['symbol']=s; a.append(x[['date','symbol','sig','fwd','fwd5']])
z=pd.concat(a)
def c(q,y):
 v=[]; n=[]
 for d,g in q.dropna(subset=['sig',y]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[y].nunique()>1:v.append(spearmanr(g.sig,g[y]).statistic);n.append(len(g))
 v=np.array(v);return len(v),round(np.mean(n),2),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6),round(np.mean(v>0),4)
print('rows',len(z),'dates',z.date.nunique(),'coverage',z.sig.notna().mean());
for y in ['fwd','fwd5']:print(y,c(z,y))
for lab,q in [('2020-22',z[z.date.dt.year<=2022]),('2023-25',z[z.date.dt.year.between(2023,2025)]),('2026',z[z.date.dt.year==2026]),('2027',z[z.date.dt.year==2027])]:print(lab,c(q,'fwd'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_1_20270715_close_reversal_1d_signal.csv',index=False)
