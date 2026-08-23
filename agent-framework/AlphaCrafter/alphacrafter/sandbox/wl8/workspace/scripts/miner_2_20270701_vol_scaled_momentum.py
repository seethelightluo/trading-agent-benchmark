import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-06-30')
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 r=x.close.pct_change(); x['sig']=x.close.pct_change(20)/(r.rolling(20).std()*np.sqrt(20)); x['f1']=x.close.shift(-1)/x.close-1; x['f5']=x.close.shift(-5)/x.close-1; x['f10']=x.close.shift(-10)/x.close-1; x['symbol']=s
 rows.append(x[['date','symbol','sig','f1','f5','f10']])
z=pd.concat(rows)
def calc(df,h):
 vals=[]; ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1:
   vals.append(spearmanr(g.sig,g[h]).statistic); ns.append(len(g))
 v=np.array(vals)
 return {'dates':len(v),'avg_n':round(np.mean(ns),2),'ic':round(v.mean(),6),'icir':round(v.mean()/v.std(ddof=1),6),'hit':round((v>0).mean(),4)}
print('END',END.date(),'rows',len(z),'valid_coverage',round(z.sig.notna().mean(),4))
for h in ['f1','f5','f10']: print(h,calc(z,h))
for label,cut in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027)]: print(label,calc(z[cut],'f1'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_2_20270701_vol_scaled_momentum_signal.csv',index=False)
