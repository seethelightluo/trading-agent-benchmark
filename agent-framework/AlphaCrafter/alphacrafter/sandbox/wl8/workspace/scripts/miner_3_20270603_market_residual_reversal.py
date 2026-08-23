import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-06-02')
parts=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date'); x['r']=x.close.pct_change(); x['symbol']=s; parts.append(x[['date','symbol','close','r']])
z=pd.concat(parts)
m=z[z.symbol=='SPX'][['date','r']].rename(columns={'r':'mr'}); z=z.merge(m,on='date',how='left')
z['covar']=z.groupby('symbol',group_keys=False).apply(lambda q:q.r.shift(1).rolling(60,min_periods=30).cov(q.mr.shift(1)),include_groups=False).reset_index(level=0,drop=True)
z['varm']=z.mr.shift(1).rolling(60,min_periods=30).var(); z['beta']=z.covar/z.varm
z['resid']=z.r-z.beta*z.mr; z['sig']=-z.groupby('symbol').resid.transform(lambda x:x.shift(1).rolling(3,min_periods=3).sum())
z['fwd1']=z.groupby('symbol').close.transform(lambda x:x.shift(-1)/x-1); z['fwd5']=z.groupby('symbol').close.transform(lambda x:x.shift(-5)/x-1)
def calc(df,h):
 a=[];ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1:a.append(spearmanr(g.sig,g[h]).statistic);ns.append(len(g))
 a=np.array(a); return len(a),len(df),round(float(np.mean(ns)),2),round(float(a.mean()),6),round(float(a.mean()/a.std(ddof=1)),6),round(float(np.mean(a>0)),4)
print('overall',calc(z,'fwd1'),'coverage',round(float(z.sig.notna().mean()),4)); print('fwd5',calc(z,'fwd5'))
for label,cut in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027)]:print(label,calc(z[cut],'fwd1'))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_3_20270603_market_residual_reversal_signal.csv',index=False)
