import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2032-08-19')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
p=pd.DataFrame(D).sort_index(); p=p[p.index<=CUT]; r=p.pct_change(); bench=r.mean(axis=1); res=r.sub(bench,axis=0)
# Equal blend of lagged 10d and 30d residual reversal, normalized by 60d total volatility.
raw=(res.rolling(10,min_periods=8).sum()+res.rolling(30,min_periods=20).sum())/2
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
sig=(-(raw/vol.replace(0,np.nan))).shift(1)
print('assets',p.shape[1],'last',p.index.max().date())
for h in [10,20,30]:
 fr=p.shift(-h)/p-1; z=[]; ns=[]; dates=[]; ranks=[]
 for d in sig.index:
  q=pd.concat([sig.loc[d],fr.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v);ns.append(len(q));dates.append(d);ranks.append(q.iloc[:,0].rank(pct=True))
 z=np.array(z); print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4),'coverage',round(np.mean(ns)/15,4),'turnover',round(np.mean([np.mean(abs(ranks[i]-ranks[i-1])) for i in range(1,len(ranks))]),6))
 for label,mask in [('2020-27',np.array(dates,dtype='datetime64[ns]')<np.datetime64('2028-01-01')),('2028-30',(np.array(dates,dtype='datetime64[ns]')>=np.datetime64('2028-01-01'))&(np.array(dates,dtype='datetime64[ns]')<np.datetime64('2031-01-01'))),('2031+',np.array(dates,dtype='datetime64[ns]')>=np.datetime64('2031-01-01')),('recent365',np.array(dates,dtype='datetime64[ns]')>=np.datetime64(CUT-pd.Timedelta(days=365)))]:
  a=z[mask]; print(' ',label,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
