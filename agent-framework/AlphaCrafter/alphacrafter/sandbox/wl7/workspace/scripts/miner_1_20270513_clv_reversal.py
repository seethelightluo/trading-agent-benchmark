import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'; f=f if os.path.exists(f) else '../persistent/index_data/'+a+'.csv'
 D[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
pd0=pd.concat({a:x['close'] for a,x in D.items()},axis=1,sort=True).ffill().loc[:'2027-05-13']
# Intraday close-location reversal, averaged over 3 sessions and risk scaled.
high=pd.concat({a:x['high'] for a,x in D.items()},axis=1,sort=True).ffill().loc[pd0.index]
low=pd.concat({a:x['low'] for a,x in D.items()},axis=1,sort=True).ffill().loc[pd0.index]
clv=((pd0-low)/(high-low).replace(0,np.nan)-0.5).rolling(3).mean()
r=pd0.pct_change(); vol=r.rolling(20).std()
f=(-clv/vol).shift(1); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',ic,'ICIR',ir,'hit',(q.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10]:
 yy=pd0.pct_change(h).shift(-h); z=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(z))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 z=q.loc[lo:hi].ic; print('regime',lo+'-'+hi,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
q.to_csv('scripts/miner_1_20270513_clv_reversal_signal.csv')
