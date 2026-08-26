import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2035-05-13']; r=px.pct_change(); b=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(b).div(b.rolling(60,min_periods=40).var(),axis=0)
res=px.pct_change(5)-beta.mul((1+b).rolling(5).apply(np.prod,raw=True)-1,axis=0)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(252); disp=r.rolling(20,min_periods=15).std().mean(axis=1)
s=(-res/(vol+1e-12)).where(disp>disp.rolling(120,min_periods=60).quantile(.60)).shift(1)
def ev(x,h):
 f=px.shift(-h)/px-1;z=[];ns=[]
 for d in x.index:
  q=pd.concat([x.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(c):z.append(c);ns.append(len(q))
 a=np.array(z);return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()
print('assets',len(A),'dates',len(px),'coverage',s.notna().mean().mean(),'active',(s!=0).mean().mean(),'turnover',s.rank(pct=True,axis=1).diff().abs().stack().mean())
for h in [1,5,10,20]:print('H',h,ev(s,h))
for st,en in [('2020','2024'),('2025','2029'),('2030','2034'),('2035-01-01','2035-05-13')]:print('regime',st,en,ev(s.loc[st:en],10))
for n in [180,500,750]:print('recent',n,ev(s.iloc[-n:],10))
s.to_csv('scripts/miner_1_20350514_beta_reversal5_signal.csv')
