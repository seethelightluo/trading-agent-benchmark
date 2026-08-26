import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(D).sort_index().loc['2020-01-01':'2034-08-05'].ffill(); r=P.pct_change(); vol=r.rolling(20,min_periods=15).std()
disp=r.std(axis=1); gate=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
F=(-(P.pct_change(5)/(vol*np.sqrt(20)+1e-12))).mul(gate,axis=0).shift(1)
def ev(h,sub=F):
 R=P.shift(-h)/P-1; a=[]; ns=[]
 for d in sub.index:
  z=pd.concat([sub.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): a.append(c);ns.append(len(z))
 q=np.asarray(a); return len(q),np.mean(ns),q.mean(),q.mean()/(q.std(ddof=1)+1e-12),(q>0).mean()
print('candidate high_disp_volnorm_reversal5 cutoff',P.index[-1].date(),'dates',len(P),'assets',len(A))
print('coverage',F.notna().mean().mean(),'active',np.mean(gate),'turnover',F.rank(axis=1,pct=True).diff().abs().stack().mean())
for h in [1,5,10,20]: print('H',h,ev(h))
for n in [180,500,750]: print('recent',n,ev(10,F.iloc[-n:]))
print('signal_rows',int(F.notna().sum().sum()))
F.to_csv('scripts/miner_1_20340807_high_disp_volnorm_reversal5_signal.csv')
