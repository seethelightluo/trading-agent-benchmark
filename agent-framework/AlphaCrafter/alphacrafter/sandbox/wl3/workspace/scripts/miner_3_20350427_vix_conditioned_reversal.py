import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c.loc[c.index<=pd.Timestamp('2035-04-27')]; r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); vx=vix.set_index('date').close.reindex(c.index).ffill(); vz=((vx-vx.rolling(120,min_periods=60).mean())/(vx.rolling(120,min_periods=60).std()+1e-12)).clip(-2,2)
s=(-r.rolling(5,min_periods=5).sum()/(vol*np.sqrt(20)+1e-12)).mul((1+0.35*vz).clip(.45,1.55),axis=0)
s.to_csv('scripts/miner_3_20350427_vix_conditioned_reversal_signal.csv',index_label='date')
f=c.pct_change(5).shift(-5); out=[]
for dt in s.index:
 ok=s.loc[dt].notna()&f.loc[dt].notna()
 if ok.sum()>=8: out.append((dt,spearmanr(s.loc[dt][ok],f.loc[dt][ok]).statistic,ok.sum()))
r2=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); a=r2.ic.dropna(); print('factor=vix_conditioned_volscaled_reversal5'); print('dates',len(r2),'instruments',15,'avg_n',r2.n.mean(),'coverage',r2.n.mean()/15); print('ic %.8f icir %.8f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),(a>0).mean()))
for h in [1,5,10,20]:
 fh=c.pct_change(h).shift(-h); q=[]
 for dt in s.index:
  ok=s.loc[dt].notna()&fh.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(s.loc[dt][ok],fh.loc[dt][ok]).statistic)
 print('horizon',h,'ic',np.nanmean(q),'n',len(q))
for name,x in [('early',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:]),('recent120',a.tail(120))]: print(name,len(x),'ic',x.mean(),'icir',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',r2.index.min().date(),r2.index.max().date())
