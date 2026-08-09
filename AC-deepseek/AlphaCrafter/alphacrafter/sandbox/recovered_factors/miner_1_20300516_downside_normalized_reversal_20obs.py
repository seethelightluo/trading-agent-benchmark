import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=P.pct_change()
# Downside-risk normalized medium-horizon reversal. PIT: trailing returns/volatility at t, forward begins t+1.
down=r.where(r<0).rolling(40,min_periods=20).std(); f=-(P.pct_change(20))/(down*np.sqrt(20)); f=f.replace([np.inf,-np.inf],np.nan)
for h in [1,3,5,10,20]:
 y=P.shift(-h)/P-1; q=[];ns=[];ds=[]
 for d in P.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(d)
 s=pd.Series(q,index=ds);print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 for n,m in [('2020-23',s.index<'2024-01-01'),('2024-27',(s.index>='2024-01-01')&(s.index<'2028-01-01')),('2028-30',s.index>='2028-01-01'),('last120',s.index>=s.index[-120])]:
  x=s[m];print(' ',n,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else 0)
print('coverage',round(f.notna().mean().mean(),4),'turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4),'range',P.index.min(),P.index.max())
# correlation against simple known comparators on pooled valid cells
for n,x in {'raw20rev':-P.pct_change(20),'raw20mom':P.pct_change(20),'volnorm5rev':-P.pct_change(5)/(r.rolling(20,min_periods=15).std()*np.sqrt(5))}.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();print('CORR',n,round(z.f.corr(z.x,method='spearman'),6),len(z))
