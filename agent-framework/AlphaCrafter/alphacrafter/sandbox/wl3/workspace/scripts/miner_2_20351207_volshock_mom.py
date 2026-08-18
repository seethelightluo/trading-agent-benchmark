import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
P=pd.DataFrame(p).sort_index(); r=P.pct_change(); v=r.rolling(20).std(); shock=(v/v.rolling(60).mean()).clip(0.5,3)
# volatility-shock conditioned short momentum: recent return weighted by abnormal volatility, lagged
f=(r.rolling(5).sum()*shock).shift(1); out=[]
for d in f.index:
 z=pd.concat([f.loc[d],(P.shift(-10)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8: out.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
print('volatility-shock conditioned momentum5; dates',len(q),'assets',15,'avg_n',q.n.mean(),'coverage',q.n.mean()/15)
print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
for a,b in [('2020','2024-12-31'),('2025','2030-12-31'),('2031','2035-12-31')]:
 x=q.loc[a:b].ic; print(a,b,len(x),x.mean(),x.mean()/x.std())
for h in [1,3,5,10,20]:
 fw=P.shift(-h)/P-1; x=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'ic',np.mean(x),'n',len(x))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20351207_volshock_mom_signal.csv',index=False)
