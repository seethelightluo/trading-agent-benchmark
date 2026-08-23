import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-06-02')
allx=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 x['overnight']=x.open/x.close.shift(1)-1
 x['fwd1']=x.close.shift(-1)/x.close-1; x['fwd5']=x.close.shift(-5)/x.close-1; x['symbol']=s
 allx.append(x[['date','symbol','overnight','fwd1','fwd5']])
z=pd.concat(allx)
# lagged cross-sectional dispersion, available before signal date (cross-asset macro state)
disp=z.assign(absret=z.overnight.abs()).groupby('date').absret.mean().shift(1)
# stronger reversal after unusually dispersed prior session
med=disp.rolling(60,min_periods=30).median(); scale=(disp/med).clip(.5,2)
z=z.merge(scale.rename('scale'),left_on='date',right_index=True,how='left'); z['sig']=-z.overnight*z.scale

def calc(df,h):
 a=[]; ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1:a.append(spearmanr(g.sig,g[h]).statistic);ns.append(len(g))
 a=np.array(a); return len(a),len(df),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)
print('overall',calc(z,'fwd1'),'coverage',z.sig.notna().mean())
print('fwd5',calc(z,'fwd5'))
for label,cut in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027)]: print(label,calc(z[cut],'fwd1'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_3_20270603_dispersion_overnight_reversal_signal.csv',index=False)
