import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-12-15'); rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 x['r3']=x.close/x.close.shift(3)-1; x['f1']=x.close.shift(-1)/x.close-1; x['symbol']=s
 rows.append(x[['date','symbol','r3','f1']])
z=pd.concat(rows,ignore_index=True)
m=pd.read_csv('../persistent/index_data/VIX.csv'); m.date=pd.to_datetime(m.date); m=m[m.date<=END].sort_values('date').set_index('date')
# lagged stress level: deviation of VIX from 60d median, clipped; source is shifted before signal use
v=(m.close/m.close.rolling(60,min_periods=30).median()-1).clip(-0.5,1.5).shift(1)
z=z.merge(v.rename('stress'),left_on='date',right_index=True,how='left')
# amplify short-term reversal only when lagged volatility stress is elevated
z['sig']=-z.r3*(1+0.8*z.stress.clip(lower=0).fillna(0))
def calc(df):
 vals=[]; ns=[]
 for d,g in df.dropna(subset=['sig','f1']).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.f1.nunique()>1:
   vals.append(spearmanr(g.sig,g.f1).statistic); ns.append(len(g))
 a=np.asarray(vals)
 return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
for name,c in [('all',z.date.notna()),('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027),('recent180',z.date>=END-pd.Timedelta(days=180)),('recent90',z.date>=END-pd.Timedelta(days=90))]:
 n,an,ic,ir,hit=calc(z[c]); print(name,'dates',n,'avg_n',round(an,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4))
print('rows',len(z),'coverage',round(z.sig.notna().mean(),4))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_2_20271216_macro_stress_reversal_signal.csv',index=False)
