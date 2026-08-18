import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];p={}
for a in A:p[a]=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close']
p=pd.DataFrame(p).sort_index().loc[:'2028-12-13'];r=p.pct_change()
for w,volw in [(3,10),(4,15),(5,15),(6,20),(7,20),(10,20),(10,30),(15,30)]:
 f=(-(p.pct_change(w))/(r.rolling(volw).std()*np.sqrt(w))).clip(-8,8); s=[];ns=[]
 for i in range(len(p)-1):
  z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
  if len(z)>=8:s.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(s);print('w volw dates cov IC ICIR hit',w,volw,len(s),round(f.notna().mean().mean(),3),round(s.mean(),5),round(s.mean()/s.std(),5),round((s>0).mean(),4))
