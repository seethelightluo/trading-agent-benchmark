import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-14'); out=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 r=x.close.pct_change(); vol=r.rolling(20,min_periods=15).std().shift(1)
 x['sig']=-r.rolling(3).sum().shift(1)/vol
 x['fwd1']=x.close.shift(-1)/x.close-1; x['fwd5']=x.close.shift(-5)/x.close-1; x['symbol']=s; out.append(x[['date','symbol','sig','fwd1','fwd5']])
z=pd.concat(out,ignore_index=True)
def calc(df,h):
 a=[]; ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1:a.append(spearmanr(g.sig,g[h]).statistic);ns.append(len(g))
 a=np.array(a)
 return len(a),int(df.sig.notna().sum()),round(float(np.mean(ns)),2),round(float(df.sig.notna().mean()),4),round(float(a.mean()),6),round(float(a.mean()/a.std(ddof=1)),6),round(float(np.mean(a>0)),4)
print('candidate=3d_vol_scaled_reversal');print('1d',calc(z,'fwd1'));print('5d',calc(z,'fwd5'))
for label,m in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027YTD',z.date.dt.year==2027)]:print(label,calc(z[m],'fwd1'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_3_20270715_vol_scaled_reversal_signal.csv',index=False)
