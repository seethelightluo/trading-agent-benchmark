import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-08-25'); rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);x=x[x.date<=END].sort_values('date');x['sig']=-x.close.pct_change(5).shift(1);x['f1']=x.close.shift(-1)/x.close-1;x['symbol']=s;rows.append(x[['date','symbol','sig','f1']])
z=pd.concat(rows)
def calc(df):
 a=[];n=[]
 for d,g in df.dropna().groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1:a.append(spearmanr(g.sig,g.f1).statistic);n.append(len(g))
 v=np.array(a);return len(v),round(np.mean(n),2),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6),round(np.mean(v>0),4)
print('range',z.date.min().date(),z.date.max().date(),'rows',len(z),'overall',calc(z),'coverage',round(z.sig.notna().mean(),4))
for q,c in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027),('last90',z.date>=END-pd.Timedelta(days=135))]:print(q,calc(z[c]))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_1_20270826_close_reversal_5d_signal.csv',index=False)
