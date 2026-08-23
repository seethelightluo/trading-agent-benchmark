import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-14')
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 x['overnight']=x.open/x.close.shift(1)-1
 x['fwd1']=x.close.shift(-1)/x.close-1; x['fwd5']=x.close.shift(-5)/x.close-1
 x['symbol']=s; rows.append(x[['date','symbol','overnight','fwd1','fwd5']])
z=pd.concat(rows,ignore_index=True)
disp=z.assign(absret=z.overnight.abs()).groupby('date').absret.mean().shift(1)
med=disp.rolling(60,min_periods=30).median(); scale=(disp/med).clip(.5,2)
z=z.merge(scale.rename('scale'),left_on='date',right_index=True,how='left'); z['sig']=-z.overnight*z.scale

def calc(df,h):
 a=[]; ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1:
   a.append(spearmanr(g.sig,g[h]).statistic); ns.append(len(g))
 a=np.array(a)
 return {'dates':len(a),'rows':int(df.sig.notna().sum()),'avg_n':round(float(np.mean(ns)),2),'coverage':round(float(df.sig.notna().mean()),4),'ic':round(float(np.mean(a)),6),'icir':round(float(np.mean(a)/np.std(a,ddof=1)),6),'hit':round(float(np.mean(a>0)),4)}
print('candidate=dispersion_scaled_overnight_reversal')
print('1d',calc(z,'fwd1')); print('5d',calc(z,'fwd5'))
for label,mask in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027YTD',z.date.dt.year==2027)]: print(label,calc(z[mask],'fwd1'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_3_20270715_dispersion_overnight_reversal_signal.csv',index=False)
