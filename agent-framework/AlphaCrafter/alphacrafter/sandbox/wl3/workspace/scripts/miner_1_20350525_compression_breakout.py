import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c.loc[c.index<=pd.Timestamp('2035-05-25')]; r=c.pct_change()
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=45).std()
s=(r.rolling(20,min_periods=20).sum()/(v20*np.sqrt(20)+1e-12))*(v60/(v20+1e-12)).clip(0.5,2.0)
s.to_csv('scripts/miner_1_20350525_compression_breakout_signal.csv',index_label='date')
for h in [1,5,10,20]:
 f=c.pct_change(h).shift(-h); q=[]
 for dt in s.index:
  ok=s.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:q.append((dt,spearmanr(s.loc[dt][ok],f.loc[dt][ok]).statistic,ok.sum()))
 z=pd.DataFrame(q,columns=['date','ic','n']).set_index('date'); a=z.ic.dropna(); ic=a.mean(); ir=ic/(a.std(ddof=1)+1e-12)*np.sqrt(len(a))
 print('horizon',h,'dates',len(z),'avg_n',z.n.mean(),'ic',round(ic,8),'icir',round(ir,4),'hit',round((a>0).mean(),4))
 if h==10:
  for nm,x in [('early',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:]),('recent120',a.tail(120))]:
   xx=x.mean(); ii=xx/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)); print(nm,len(x),'ic',round(xx,8),'icir',round(ii,4))
  print('coverage',z.n.mean()/15,'turnover',s.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',z.index.min().date(),z.index.max().date())
