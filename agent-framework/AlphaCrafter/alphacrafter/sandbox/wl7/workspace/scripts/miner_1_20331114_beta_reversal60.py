import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
px=pd.DataFrame(D).sort_index().loc['2020-01-01':'2033-11-13']; r=px.pct_change(); b=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(b).div(b.rolling(60,min_periods=40).var(),axis=0)
res=px.pct_change(10)-beta.mul((1+b).rolling(10).apply(np.prod,raw=True)-1,axis=0)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(252)
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
s=-res/(vol+1e-12); gate=disp>disp.rolling(120,min_periods=60).quantile(.60); s=s.where(gate,0).shift(1)
def ev(h):
 f=px.shift(-h)/px-1; z=[]; ns=[]
 for d in s.index:
  q=pd.concat([s.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(c):z.append(c);ns.append(len(q))
 a=np.array(z);return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()
print('60pct beta residual reversal',len(A),len(px),px.index[-1].date(),'coverage',s.notna().mean().mean(),'active',(s!=0).mean().mean(),'turnover',s.rank(pct=True,axis=1).diff().abs().stack().mean())
for h in [1,5,10,20]:print(h,ev(h))
for n in [180,500,750]:
 old=s;s=s.iloc[-n:];print('recent',n,ev(10));s=old
s.to_csv('scripts/miner_1_20331114_beta_reversal60_signal.csv')
