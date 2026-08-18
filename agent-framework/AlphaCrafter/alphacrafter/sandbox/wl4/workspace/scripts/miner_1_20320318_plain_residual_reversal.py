import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];a={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);a[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(a).sort_index();r=p.pct_change();
for h in [5,10,20]:
 out=[];fs=[]
 for i in range(65,len(p)-h):
  q=r.iloc[i-4:i+1].sum();f=-(q-q.median());y=p.iloc[i+h]/p.iloc[i]-1;z=pd.concat([f,y],axis=1).dropna()
  if len(z)>=8:out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);fs.append(f)
 x=np.array(out);print('H',h,'dates',len(x),'avgN',np.mean([len(pd.concat([fs[j],pd.Series(index=[])],axis=1)) for j in []]) if False else 15,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(len(x)),'hit',np.mean(x>0))
 for n in [250,500]:
  q=x[-n:];print(' recent',n,q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
 print('turnover',np.mean([np.mean(abs(fs[j].rank(pct=True)-fs[j-10].rank(pct=True))) for j in range(10,len(fs),10)]))
