import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-14')
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date').copy()
 x['sig']=-(x.open/x.close.shift(1)-1); x['fwd']=x.close.shift(-1)/x.close-1; x['fwd5']=x.close.shift(-5)/x.close-1; x['symbol']=s
 rows.append(x[['date','symbol','sig','fwd','fwd5']])
z=pd.concat(rows,ignore_index=True)
def calc(q,y):
 vals=[]; ns=[]
 for d,g in q.dropna(subset=['sig',y]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[y].nunique()>1:
   vals.append(spearmanr(g.sig,g[y]).statistic); ns.append(len(g))
 v=np.array(vals); return {'dates':len(v),'mean_n':round(float(np.mean(ns)),2),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float(np.mean(v>0)),4)} if len(v)>1 else {}
print('rows',len(z),'dates',z.date.nunique(),'coverage',round(z.sig.notna().mean(),6))
for h in ['fwd','fwd5']: print(h,calc(z,h))
for lab,q in [('2020-22',z[z.date.dt.year<=2022]),('2023-25',z[z.date.dt.year.between(2023,2025)]),('2026',z[z.date.dt.year==2026]),('2027',z[z.date.dt.year==2027])]: print(lab,calc(q,'fwd'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_1_20270715_overnight_gap_reversal_signal.csv',index=False)
