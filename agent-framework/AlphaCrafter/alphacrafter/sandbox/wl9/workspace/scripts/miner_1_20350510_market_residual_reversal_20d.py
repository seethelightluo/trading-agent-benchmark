import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
prices=pd.DataFrame({s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index().loc[:'2035-05-09']; r=prices.pct_change(); m=r.mean(axis=1)
rm=(r.mul(m,axis=0)).rolling(60,min_periods=40).mean(); beta=(rm-r.rolling(60,min_periods=40).mean().mul(m.rolling(60,min_periods=40).mean(),axis=0))/(m.rolling(60,min_periods=40).var()+1e-8)
a20=prices.pct_change(20); mm=(1+m).rolling(20).apply(np.prod,raw=True)-1
vol=r.rolling(40,min_periods=25).std()*np.sqrt(252); f=(- (a20-beta*mm)/(vol+0.005)).clip(-5,5).shift(1)
def calc(h):
 vals=[]; dates=[]; ns=[]; fw=prices.pct_change(h)
 for i in range(len(prices)-h):
  z=pd.concat([f.iloc[i],fw.iloc[i+h]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(prices.index[i]); ns.append(len(z))
 x=pd.Series(vals,index=pd.DatetimeIndex(dates)); return x,np.mean(ns)
print('universe',len(U),'dates',prices.index.min().date(),prices.index.max().date())
for h in [5,10,20,40,60]:
 x,an=calc(h); print('H',h,'dates',len(x),'avgN',round(an,2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round((x>0).mean(),4))
x,_=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print('REG',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover_proxy',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350510_market_residual_reversal_20d_signal.csv',index=False)
