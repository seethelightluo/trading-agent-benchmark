import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 d=get_stock_daily_data(s,6000)
 if d is None or len(d)==0:d=get_index_daily_data(s,6000)
 return d.set_index('date')['close'].astype(float) if d is not None and len(d) else None
P=pd.DataFrame({s:L(s) for s in U}).sort_index().ffill().loc[:pd.Timestamp('2035-10-24')]
r=P.pct_change(); m=P/P.shift(20)-1; v=r.rolling(60).std(); cs=m.sub(m.median(axis=1),axis=0); disp=m.std(axis=1); gate=disp>disp.rolling(120,min_periods=60).median(); F=(-cs/(v+1e-8)).where(gate,0).shift(1)
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; a=[]; ns=[]
 for d in F.index:
  ok=(F.loc[d]!=0)&F.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   q=F.loc[d,ok].corr(fw.loc[d,ok],method='spearman')
   if pd.notna(q):a.append(q);ns.append(ok.sum())
 a=pd.Series(a);print('h=%d dates=%d avgN=%.3f IC=%.8f ICIR=%.8f hit=%.4f'%(h,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(len(a)),(a>0).mean()))
print('coverage',F.notna().sum().sum()/(len(F)*15),'active',(F!=0).any(axis=1).sum(),'rows',len(F),'end',P.index.max().date())
F.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20351025_dispersion_gated_reversal_signal.csv',index=False)
