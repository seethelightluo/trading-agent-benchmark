import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
end=pd.Timestamp('2028-04-19'); dates=sorted(set.intersection(*[set(x.index) for x in D.values()])); dates=[d for d in dates if d<=end]
out=[]
for t in dates:
 vals=[]; fw=[]
 for s in U:
  x=D[s]; j=x.index.get_loc(t)
  if j>=20 and j+10<len(x):
   r=x.close.iloc[j]/x.close.iloc[j-20]-1; pos=(np.diff(x.close.iloc[j-20:j+1])>0).sum()/20; vals.append(r*(.5+.5*pos)); fw.append(x.close.iloc[j+10]/x.close.iloc[j]-1)
 if len(vals)>=8:out.append(spearmanr(vals,fw).statistic)
a=np.array(out); print('dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean());print('recent250',a[-250:].mean(),a[-250:].mean()/a[-250:].std(ddof=1))
