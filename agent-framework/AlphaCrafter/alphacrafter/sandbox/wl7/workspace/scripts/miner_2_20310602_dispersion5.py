import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2031-06-02')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]; c=x.close.astype(float); r=c.pct_change()
 D[s]=pd.DataFrame({'r5':c.pct_change(5),'v':r.rolling(20).std(),'f1':c.shift(-1)/c-1,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1,'f20':c.shift(-20)/c-1})
common=sorted(set.intersection(*[set(v.index) for v in D.values()])); rows=[]
for dt in common:
 z=pd.DataFrame({s:{k:D[s].loc[dt,k] for k in ['r5','v','f1']} for s in U}).T.dropna()
 # high cross-sectional dispersion relative to its own cross-sectional median absolute move
 if len(z)>=8 and z.r5.abs().median()>0 and z.r5.abs().mean()>1.35*z.r5.abs().median():
  q=pd.DataFrame({'f':-(z.r5-z.r5.median())/(z.v*np.sqrt(5)),'y':z.f1}).dropna()
  if len(q)>=8: rows.append((dt,spearmanr(q.f,q.y).statistic,len(q)))
a=pd.Series([x[1] for x in rows]); print('dates',len(a),'assets',len(U),'avgN %.2f'%np.mean([x[2] for x in rows]),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
for h in [5,10,20]:
 q=[]
 for dt in common:
  z=pd.DataFrame({s:{'r5':D[s].loc[dt,'r5'],'v':D[s].loc[dt,'v'],'y':D[s].loc[dt,'f'+str(h)]} for s in U}).T.dropna()
  if len(z)>=8 and z.r5.abs().median()>0 and z.r5.abs().mean()>1.35*z.r5.abs().median(): q.append(spearmanr(-(z.r5-z.r5.median())/(z.v*np.sqrt(5)),z.y).statistic)
 q=pd.Series(q);print('h',h,'IC %.8f ICIR %.8f n %d'%(q.mean(),q.mean()/q.std(),len(q)))
print('coverage %.6f period %s %s'%(len(a)/len(common),common[0],common[-1]))
print('regimes',*[round(a.iloc[i:j].mean(),6) for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
out=[]
for dt in common:
 z=pd.DataFrame({s:{'r5':D[s].loc[dt,'r5'],'v':D[s].loc[dt,'v']} for s in U}).T.dropna()
 if len(z)>=8 and z.r5.abs().median()>0 and z.r5.abs().mean()>1.35*z.r5.abs().median():
  for s,val in (-(z.r5-z.r5.median())/(z.v*np.sqrt(5))).items():out.append((dt,s,val))
pd.DataFrame(out,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20310602_dispersion5_signal.csv',index=False)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20310602_dispersion5_ic.csv',index=False)
