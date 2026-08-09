import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}
idx=pd.Index(sorted(set().union(*[set(x.index) for x in P.values()])))
c=pd.DataFrame({a:P[a].reindex(idx).ffill() for a in A}); r=c.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(idx).ffill(); vr=v.pct_change()
# One interpretable idea: asset relative performance on high-VIX relief days, normalized by own volatility.
relief=(vr<0)&(v>v.rolling(60,min_periods=40).quantile(.70))
den=r.rolling(20,min_periods=12).std()
sig=r.where(relief).rolling(60,min_periods=12).mean()/den
sig=sig.sub(sig.median(axis=1),axis=0)
print('visible',idx[idx<='2031-08-06'].max().date(),'dates',len(idx),'assets',len(A),'relief_days',int(relief.sum()),'coverage',round(sig.notna().mean().mean(),4))
def run(H,mask=None):
 f=c.shift(-H)/c-1; vals=[];ns=[]
 for i in range(len(idx)-H):
  if mask is not None and not bool(mask[i]):continue
  z=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(vals);return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)
for h in [1,5,10,20]:
 n,m,ic,ir,hit=run(h);print('H',h,'dates',n,'meanN',round(m,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4))
for lab,mask in [('2020-23',idx<'2024-01-01'),('2024-27',(idx>='2024-01-01')&(idx<'2028-01-01')),('2028-30',(idx>='2028-01-01')&(idx<'2031-01-01')),('2031',idx>='2031-01-01')]:
 n,m,ic,ir,hit=run(10,mask);print(lab,'dates',n,'IC',round(ic,6),'ICIR',round(ir,6))
print('turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
for name,q in [('invvol',-den),('mom20',c.pct_change(20))]:
 z=pd.concat([sig.stack(),q.stack()],axis=1).dropna();print('corr',name,round(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,4),len(z))
