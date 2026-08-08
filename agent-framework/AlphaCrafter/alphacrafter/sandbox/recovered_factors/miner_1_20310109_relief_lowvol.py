import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(a):
 f=glob.glob('../persistent/stock_data/'+a+'.csv'); d=pd.read_csv(f[0]); d.date=pd.to_datetime(d.date); return d.set_index('date').close.astype(float)
p=pd.concat({a:L(a) for a in A},axis=1).sort_index(); r=np.log(p).diff();
def M(a):
 d=pd.read_csv('../persistent/index_data/'+a+'.csv');d.date=pd.to_datetime(d.date);return d.set_index('date').close
v=np.log(M('VIX')).diff().reindex(p.index); x=np.log(M('DXY')).diff().reindex(p.index)
mask=(v.rolling(5).sum()<0)&(x.rolling(5).sum()<0)
# low realized volatility, cross-section demeaned, during macro relief
vol=r.rolling(20).std(); f=(-vol).sub((-vol).mean(axis=1),axis=0).where(mask)
for h in [1,5,10,20]:
 z=[]
 for i in range(len(p)-h):
  if not mask.iloc[i]:continue
  q=pd.concat([f.iloc[i],r.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=np.array(z);print('H',h,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'dates',len(z))
print('coverage',f.notna().sum().sum()/f.size,'relief dates',mask.sum())
