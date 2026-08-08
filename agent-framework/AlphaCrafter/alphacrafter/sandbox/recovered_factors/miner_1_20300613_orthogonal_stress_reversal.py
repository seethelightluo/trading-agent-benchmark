import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for a in A}).sort_index(); r=P.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill(); vr=vix.pct_change(); stress=(vr.rolling(20,min_periods=12).mean().clip(lower=0))
# orthogonal stress-conditioned reversal: raw 3d reversal, residualized cross-sectionally against 20d trend and 20d volatility
raw=-P.pct_change(3); trend=P.pct_change(20); vol=r.rolling(20,min_periods=15).std(); f=pd.DataFrame(index=P.index,columns=A,dtype=float)
for d in P.index:
 z=pd.DataFrame({'raw':raw.loc[d],'trend':trend.loc[d],'vol':vol.loc[d]}).dropna()
 if len(z)>=8:
  X=np.column_stack([np.ones(len(z)),z[['trend','vol']].values]); b=np.linalg.lstsq(X,z.raw.values,rcond=None)[0]; f.loc[d,z.index]=(z.raw-X@b)*stress.loc[d]
for h in [1,3,5,10,20]:
 y=P.shift(-h)/P-1; q=[];ns=[];ds=[]
 for d in P.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(d)
 s=pd.Series(q,index=ds); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 for lab,m in [('2020-23',s.index<'2024-01-01'),('2024-27',(s.index>='2024-01-01')&(s.index<'2028-01-01')),('2028-30',s.index>='2028-01-01'),('last120',s.index>=s.index[-120])]:
  x=s[m]; print(lab,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else 0)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
# audit against interpretable library proxy signals, pooled Spearman
libs={'raw3rev':-P.pct_change(3),'raw20mom':P.pct_change(20),'risktrend':trend/vol,'vol':-vol,'raw5rev':-P.pct_change(5),'kurt':-r.rolling(40,min_periods=30).kurt(),'es':-r.rolling(40,min_periods=30).quantile(.2),'vixbeta':r.rolling(20,min_periods=15).cov(vr)/vr.rolling(20,min_periods=15).var()}
for n,x in libs.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); print('CORR',n,round(z.f.corr(z.x,method='spearman'),6),len(z))
print('range',P.index.min(),P.index.max())
