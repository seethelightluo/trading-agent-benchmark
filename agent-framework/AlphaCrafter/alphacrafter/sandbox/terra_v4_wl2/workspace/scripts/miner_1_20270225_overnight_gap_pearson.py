import pandas as pd,numpy as np
from scipy.stats import pearsonr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-02-25');R=[]
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@end').sort_values('date');d['s']=- (d.open/d.close.shift(1)-1);d['r']=d.close.shift(-5)/d.close-1;d['a']=a;R.append(d[['date','a','s','r']])
x=pd.concat(R);z=[]
for dt,g in x.groupby('date'):
 g=g.dropna();
 if len(g)>=8 and g.s.nunique()>1 and g.r.nunique()>1:z.append(pearsonr(g.s,g.r).statistic)
z=np.array(z);print(len(z),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0))
print('nonfuture range',x.date.min(),x.date.max())
