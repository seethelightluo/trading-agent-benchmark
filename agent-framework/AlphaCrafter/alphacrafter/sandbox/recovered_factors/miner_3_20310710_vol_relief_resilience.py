import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
idx=None; P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); P[a]=d; idx=d.index if idx is None else idx.union(d.index)
idx=idx.sort_values(); c=pd.DataFrame({a:P[a] for a in A},index=idx).ffill(); r=c.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].reindex(idx).ffill(); vr=v.pct_change()
# volatility relief: VIX falls following elevated VIX; score asset return on relief days, normalized by its 20d vol
relief=(vr<0)&(v>v.rolling(60,min_periods=40).quantile(.70))
num=(r.where(relief)).rolling(60,min_periods=12).mean(); den=r.rolling(20,min_periods=12).std(); sig=(num/den)
# remove common market component and center cross section
m=r.mean(axis=1); ms=m.rolling(60,min_periods=20).mean(); mb=m.rolling(60,min_periods=20).cov(m) # unused
sig=sig.sub(sig.median(axis=1),axis=0)
print('dates',len(idx),'assets',len(A),'relief_days',int(relief.sum()),'coverage',round(sig.notna().mean().mean(),4))
def run(H,mask=None):
 f=c.shift(-H)/c-1; vals=[]; ns=[]; ds=[]
 for i,dt in enumerate(idx[:-H]):
  if mask is not None and not bool(mask.iloc[i]): continue
  z=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 x=np.array(vals); return len(x),np.mean(ns),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),pd.Series(x,index=ds)
for h in [1,5,10,20]:
 n,m,ic,ir,hit,s=run(h);print('H',h,'dates',n,'meanN',round(m,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4))
for lab,mask in [('2020-23',idx<'2024-01-01'),('2024-27',(idx>='2024-01-01')&(idx<'2028-01-01')),('2028-30',(idx>='2028-01-01')&(idx<'2031-01-01')),('2031',idx>='2031-01-01')]:
 n,m,ic,ir,hit,s=run(10,mask);print(lab,n,round(ic,6),round(ir,6))
# signal turnover
print('turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
for name,q in [('invvol',-r.rolling(20,min_periods=12).std()),('mom20',c.pct_change(20)),('vixrelief',r.where(relief).rolling(60,min_periods=12).mean()/den)]:
 z=pd.concat([sig.stack(),q.stack()],axis=1).dropna();print('corr',name,round(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,4),len(z))
# recent
n,m,ic,ir,hit,s=run(10); z=s.iloc[-120:];print('latest120',len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
