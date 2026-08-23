import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); D[a]=x.close.astype(float)
p=pd.DataFrame(D).sort_index(); cutoff=pd.Timestamp('2032-04-28'); p=p.loc[:cutoff]; r=p.pct_change(); neg=r.where(r<0)
f=-(p.shift(1)/p.shift(21)-1)/(neg.shift(1).rolling(60,min_periods=30).std()*np.sqrt(20)+1e-12)
fr=p.shift(-5)/p-1; I=[];N=[];T=[];prev=None
for dt in p.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:I.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);N.append(len(z))
 rk=f.loc[dt].rank(pct=True)
 if prev is not None:
  q=pd.concat([rk,prev],axis=1).dropna();T.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
 prev=rk
I=np.array(I); print({'cutoff':str(cutoff.date()),'dates':len(I),'avg_n':round(np.mean(N),2),'coverage':round(np.mean(N)/15,4),'ic':round(np.mean(I),6),'icir':round(np.mean(I)/np.std(I,ddof=1),4),'hit':round(np.mean(I>0),4),'turnover':round(np.nanmean(T),4)})
print(pd.DataFrame({'ic':I}).assign(year=0).head(1).to_string(index=False))
