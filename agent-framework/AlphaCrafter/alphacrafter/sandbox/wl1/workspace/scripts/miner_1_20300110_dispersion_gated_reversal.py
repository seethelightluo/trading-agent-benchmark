import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-01-10')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cutoff]
r=P.pct_change(); base=(-(P/P.shift(5)-1)/(r.rolling(20,min_periods=18).std()+1e-8)).shift(1)
# Daily cross-sectional dispersion, lagged before use.
disp=r.std(axis=1).shift(1)
for q in [0.5,0.7,0.8]:
 threshold=disp.rolling(120,min_periods=60).quantile(q)
 gate=(disp>threshold).astype(float)
 f=base*(1+gate)
 fr=P.shift(-20)/P-1; vals=[]; dates=[]; n=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(dt);n.append(len(z))
 a=np.array(vals); print('q',q,'dates',len(a),'avgN',np.mean(n),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(len(a)),'hit',(a>0).mean())
 for label,lo in [('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
  x=a[np.array(dates)>=lo]; print(label,len(x),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(len(x)))
 print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f=base;fr=P.shift(-20)/P-1; a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
a=np.array(a);print('BASE',len(a),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(len(a)))
