import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=P.pct_change()
# reversal after a volatility shock: recent 5d return, scaled by vol expansion (5d/40d); high score means buy beaten down during vol shock
v5=r.rolling(5,min_periods=4).std(); v40=r.rolling(40,min_periods=30).std(); f=-P.pct_change(5)*(v5/v40)
for h in [1,3,5,10,20]:
 y=P.shift(-h)/P-1; ic=[];ns=[];ds=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 s=pd.Series(ic,index=ds); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 for n,m in [('2024-27',(s.index>='2024-01-01')&(s.index<'2028-01-01')),('2028-30',s.index>='2028-01-01'),('last120',s.index>=s.index[-120])]:
  q=s[m];print(' ',n,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else 0)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
