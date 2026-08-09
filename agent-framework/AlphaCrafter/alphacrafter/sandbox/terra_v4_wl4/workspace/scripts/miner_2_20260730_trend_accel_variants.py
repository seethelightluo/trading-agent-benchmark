import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U};p=pd.concat(D,axis=1).sort_index();r=np.log(p).diff()
for a,b in [(5,20),(10,30),(10,40),(15,45),(20,60),(30,90),(40,120)]:
 f=(p.pct_change(a)-p.pct_change(b))/(r.rolling(b).std()*np.sqrt(b)); out=[]; ns=[]
 for i in range(b,len(p)-1):
  z=pd.concat([f.iloc[i].rename('f'),p.iloc[i+1].div(p.iloc[i]).sub(1).rename('y')],axis=1).dropna()
  if len(z)>=8:out.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.array(out);print(a,b,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5),'hit',round(np.mean(q>0),4),'cov',round(np.mean(ns)/15,4))
