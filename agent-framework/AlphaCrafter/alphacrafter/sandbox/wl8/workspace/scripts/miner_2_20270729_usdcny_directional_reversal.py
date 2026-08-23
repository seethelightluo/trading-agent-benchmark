import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-28'); rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date'); x['r3']=x.close/x.close.shift(3)-1; x['f1']=x.close.shift(-1)/x.close-1; x['symbol']=s; rows.append(x[['date','symbol','r3','f1']])
z=pd.concat(rows); m=pd.read_csv('../persistent/index_data/USDCNY.csv'); m.date=pd.to_datetime(m.date); m=m[m.date<=END].sort_values('date').set_index('date'); move=m.close.pct_change(5); vol=move.rolling(60,min_periods=30).std(); state=(move/vol).shift(1).clip(-2,2); z=z.merge(state.rename('st'),left_on='date',right_index=True,how='left'); z['sig']=-z.r3*(1+0.5*z.st)
def calc(df):
 a=[]; ns=[]
 for d,g in df.dropna(subset=['sig','f1']).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.f1.nunique()>1: a.append(spearmanr(g.sig,g.f1).statistic); ns.append(len(g))
 a=np.array(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
for name,c in [('all',z.date<=END),('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027),('last90',z.date>=END-pd.Timedelta(days=90))]: print(name,calc(z[c]))
print('coverage',z.sig.notna().mean()); z[['date','symbol','sig']].dropna().to_csv('scripts/miner_2_20270729_usdcny_directional_reversal_signal.csv',index=False)
