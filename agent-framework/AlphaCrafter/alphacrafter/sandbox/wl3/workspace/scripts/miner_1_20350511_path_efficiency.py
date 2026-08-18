import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c.loc[c.index<=pd.Timestamp('2035-05-11')]; r=c.pct_change()
# Path-efficiency momentum: signed 30d move divided by total absolute daily movement, with volatility normalization.
# This rewards persistent directional trends and penalizes choppy paths.
move=r.rolling(30,min_periods=25).sum(); path=r.abs().rolling(30,min_periods=25).sum(); vol=r.rolling(60,min_periods=40).std()
s=(move/(path+1e-12))*(move/(vol*np.sqrt(30)+1e-12))
s.to_csv('scripts/miner_1_20350511_path_efficiency_momentum_signal.csv',index_label='date')
rows=[]
for dt in s.index:
 f=c.pct_change(10).shift(-10); ok=s.loc[dt].notna()&f.loc[dt].notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(s.loc[dt][ok],f.loc[dt][ok]).statistic,ok.sum()))
r2=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=r2.ic.dropna()
print('factor=path_efficiency_momentum30');print('dates',len(r2),'instruments',15,'avg_n',r2.n.mean(),'coverage',r2.n.mean()/15)
print('ic %.8f icir %.8f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),(a>0).mean()))
for h in [1,5,10,20]:
 fh=c.pct_change(h).shift(-h); q=[]
 for dt in s.index:
  ok=s.loc[dt].notna()&fh.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(s.loc[dt][ok],fh.loc[dt][ok]).statistic)
 print('horizon',h,'ic',np.nanmean(q),'n',len(q))
for name,x in [('early',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:]),('recent120',a.tail(120))]: print(name,len(x),'ic',x.mean(),'icir',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',r2.index.min().date(),r2.index.max().date())
