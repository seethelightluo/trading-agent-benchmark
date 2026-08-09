import pandas as pd,numpy as np
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
px=pd.DataFrame(D).sort_index(); r=px.pct_change();
# smooth medium momentum, volatility adjusted, lagged one day
f=px.pct_change(20)/(r.rolling(40,min_periods=30).std()*np.sqrt(20))
# cross-sectional demean reduces common beta, then rank
f=f.sub(f.mean(axis=1),axis=0)
for h in [1,3,5,10,20]:
 y=px.shift(-h)/px-1; vals=[]; ns=[]; ds=[]
 for dt in px.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 for name,mask in [('2024-27',(s.index>='2024-01-01')&(s.index<'2028-01-01')),('2028-30',s.index>='2028-01-01'),('last120',s.index>=s.index[-120])]:
  q=s[mask];print(' ',name,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else 0)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'range',px.index.min(),px.index.max())
