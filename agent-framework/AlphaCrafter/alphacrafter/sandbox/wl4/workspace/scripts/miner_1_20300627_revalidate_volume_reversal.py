import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-06-26'); root='../persistent/stock_data'; px={}; vv={}
for s in U:
 d=pd.read_csv(os.path.join(root,s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); px[s]=d['close'].astype(float); vv[s]=d['volume'].astype(float)
p=pd.DataFrame(px).sort_index().loc[:cut]; v=pd.DataFrame(vv).reindex(p.index); r=p.pct_change()
# signal uses completed session t-1 only; forward return starts from close t-1 to t-1+H
for H in [5,10,20]:
 out=[]
 for i in range(26,len(p)-H):
  j=i-1
  sig=(-(p.iloc[j]/p.iloc[j-5]-1)*(v.iloc[j]/v.iloc[j-20:j].mean()).replace([np.inf,-np.inf],np.nan))
  fwd=p.iloc[j+H]/p.iloc[j]-1
  z=pd.concat([sig,fwd],axis=1).dropna()
  if len(z)>=8: out.append((p.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 x=pd.DataFrame(out,columns=['date','ic','n'])
 for name,y in [('full',x),('recent',x.tail(261)),('early',x.iloc[:len(x)//3]),('mid',x.iloc[len(x)//3:2*len(x)//3]),('late',x.iloc[2*len(x)//3:])]:
  m=y.ic.mean(); sd=y.ic.std(ddof=1); ir=m/sd*np.sqrt(len(y))
  print(H,name,'dates',len(y),'avgN',round(y.n.mean(),2),'IC',round(m,6),'ICIR',round(ir,6),'hit',round((y.ic>0).mean(),4))
print('cutoff',p.index[-1].date(),'assets',len(U))
