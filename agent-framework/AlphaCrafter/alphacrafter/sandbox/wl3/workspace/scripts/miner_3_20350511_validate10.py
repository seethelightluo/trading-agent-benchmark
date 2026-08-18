import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data';p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'));d['date']=pd.to_datetime(d.date);p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index();c=c.loc[c.index<=pd.Timestamp('2035-05-11')];r=c.pct_change();down=r.where(r<0,0).rolling(30,min_periods=20).std();s=c.pct_change(20)/(down*np.sqrt(20)+1e-12);f=c.pct_change(10).shift(-10);rows=[]
for dt in s.index:
 ok=s.loc[dt].notna()&f.loc[dt].notna()
 if ok.sum()>=8:rows.append((dt,spearmanr(s.loc[dt][ok],f.loc[dt][ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');a=x.ic.dropna();print('dates',len(a),'mean',a.mean(),'icir',a.mean()/a.std(ddof=1)*np.sqrt(len(a)),'hit',(a>0).mean(),'coverage',x.n.mean()/15)
print('recent120',a.tail(120).mean(),a.tail(120).mean()/a.tail(120).std(ddof=1)*np.sqrt(120))
