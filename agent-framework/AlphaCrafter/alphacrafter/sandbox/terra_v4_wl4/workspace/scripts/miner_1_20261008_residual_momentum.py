import numpy as np, pandas as pd
from pathlib import Path
root=Path('../persistent'); end=pd.Timestamp('2026-10-07'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(n,macro=False):
 p=root/('index_data' if macro else 'stock_data')/(n+'.csv'); return pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')['close'].astype(float).loc[:end]
px=pd.concat({s:load(s) for s in syms},axis=1); spx=load('SPX'); dxy=load('DXY',True); r=px.pct_change(); rs=spx.pct_change(); rd=dxy.pct_change()
def beta(x,z): return x.rolling(60,min_periods=45).cov(z).div(z.rolling(60,min_periods=45).var())
b1=r.apply(lambda x: beta(x,rs)); b2=r.apply(lambda x: beta(x,rd)); f=px.pct_change(20)-b1.mul(spx.pct_change(20),axis=0)-b2.mul(dxy.pct_change(20),axis=0); f=f.replace([np.inf,-np.inf],np.nan)
def evaluate(h):
 y=px.shift(-h).div(px)-1; out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
q=evaluate(1); print('factor=20d SPX+DXY residual momentum; dates',len(q),'mean names',q.n.mean(),'coverage',q.n.sum()/(len(q)*15)); print('IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=q.loc[a:b]; print(a+'-'+b,'dates',len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
for h in [5,10]:
 z=evaluate(h); print('horizon',h,'dates',len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std()))
ranks=f.rank(axis=1,pct=True); print('rank turnover',ranks.loc[ranks.notna().sum(axis=1)>=8].diff().abs().mean(axis=1).mean()); print('validation_end',q.index.max().date()); f.to_csv('scripts/miner_1_20261008_residual_momentum_signal.csv',index_label='date')
