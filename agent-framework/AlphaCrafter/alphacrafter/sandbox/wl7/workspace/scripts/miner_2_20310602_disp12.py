import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2031-06-02'); T=1.20
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]; c=x.close.astype(float); r=c.pct_change()
 D[s]=pd.DataFrame({'r3':c.pct_change(3),'v':r.rolling(20).std(),'f1':c.shift(-1)/c-1,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1,'f20':c.shift(-20)/c-1})
common=sorted(set.intersection(*[set(v.index) for v in D.values()])); rows=[]
for dt in common:
 z=pd.DataFrame({s:{k:D[s].loc[dt,k] for k in ['r3','v','f1']} for s in U}).T.dropna()
 if len(z)>=8 and z.r3.abs().median()>0 and z.r3.abs().mean()>T*z.r3.abs().median():
  q=pd.DataFrame({'f':-(z.r3-z.r3.median())/(z.v*np.sqrt(3)),'y':z.f1}).dropna()
  if len(q)>=8: rows.append((dt,spearmanr(q.f,q.y).statistic,len(q)))
a=pd.Series([x[1] for x in rows]); print('threshold',T,'dates',len(a),'avgN %.2f'%np.mean([x[2] for x in rows]),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()),'coverage',len(a)/len(common))
for h in [5,10,20]:
 q=[]
 for dt in common:
  z=pd.DataFrame({s:{'r3':D[s].loc[dt,'r3'],'v':D[s].loc[dt,'v'],'y':D[s].loc[dt,'f'+str(h)]} for s in U}).T.dropna()
  if len(z)>=8 and z.r3.abs().median()>0 and z.r3.abs().mean()>T*z.r3.abs().median(): q.append(spearmanr(-(z.r3-z.r3.median())/(z.v*np.sqrt(3)),z.y).statistic)
 q=pd.Series(q);print('h',h,'IC %.8f ICIR %.8f n %d'%(q.mean(),q.mean()/q.std(),len(q)))
out=[]
for dt in common:
 z=pd.DataFrame({s:{'r3':D[s].loc[dt,'r3'],'v':D[s].loc[dt,'v']} for s in U}).T.dropna()
 if len(z)>=8 and z.r3.abs().median()>0 and z.r3.abs().mean()>T*z.r3.abs().median():
  for s,val in (-(z.r3-z.r3.median())/(z.v*np.sqrt(3))).items():out.append((dt,s,val))
pd.DataFrame(out,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20310602_disp12_signal.csv',index=False)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20310602_disp12_ic.csv',index=False)
