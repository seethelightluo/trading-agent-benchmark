import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-11-03')
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date'); x['r3']=x.close/x.close.shift(3)-1; x['f1']=x.close.shift(-1)/x.close-1; x['symbol']=s; rows.append(x[['date','symbol','r3','f1']])
z=pd.concat(rows)
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v[v.date<=END].sort_values('date').set_index('date')
# lagged VIX shock: emphasize reversal after unusually large completed 5d VIX rise
chg=v.close.pct_change(5); base=chg.rolling(60,min_periods=30).std(); shock=(chg/base).shift(1).clip(-2,2)
z=z.merge(shock.rename('shock'),left_on='date',right_index=True,how='left'); z['sig']=-z.r3*(1+0.5*z.shock.clip(lower=0))
def calc(df):
 vals=[]; ns=[]
 for d,g in df.dropna(subset=['sig','f1']).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1: vals.append(spearmanr(g.sig,g.f1).statistic); ns.append(len(g))
 a=np.asarray(vals); return len(a),round(float(np.mean(ns)),2),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean())
print('end',z.date.max().date(),'overall',calc(z)); y=z.date.dt.year
for q,m in [('2020-22',y<=2022),('2023-25',y.between(2023,2025)),('2026',y==2026),('2027',y==2027),('last180',z.date>=END-pd.Timedelta(days=180))]: print(q,calc(z[m]))
print('coverage',round(z.sig.notna().mean(),4),'rows',len(z),'dates',z.date.nunique(),'avg universe',round(z.groupby('date').sig.apply(lambda x:x.notna().sum()).mean(),2))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_1_20271104_vix_stress_reversal_signal.csv',index=False)
