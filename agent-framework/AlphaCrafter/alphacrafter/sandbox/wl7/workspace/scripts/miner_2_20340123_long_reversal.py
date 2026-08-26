import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=2200) for s in U};px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill();r=np.log(px).diff()
# medium/long residual momentum, inverse direction, risk scaled; lagged
m=r.rolling(60).sum(); v=r.rolling(40).std(); f=-(m.sub(m.mean(axis=1),axis=0))/(v+1e-12); f=f.shift(1)
for h in [1,5,10,20]:
 fr=np.log(px.shift(-h)/px); vals=[];ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8:
   q=f.loc[dt][ok].corr(fr.loc[dt][ok],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(ok.sum())
 z=pd.Series(vals).dropna();print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_2_20340123_long_reversal_signal.csv',index=False)
