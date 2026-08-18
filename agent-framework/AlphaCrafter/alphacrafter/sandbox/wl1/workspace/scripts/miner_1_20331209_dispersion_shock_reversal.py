import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'; end=pd.Timestamp('2033-12-09')
px={}
for s in U:
 d=pd.read_csv(os.path.join(P,s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close
C=pd.DataFrame(px).sort_index().loc[:end]; R=C.pct_change(); D=R.std(axis=1); dz=(D-D.rolling(60,min_periods=30).mean())/(D.rolling(60,min_periods=30).std()+1e-12)
rv=R.rolling(20,min_periods=15).std(); r5=C.pct_change(5)
mult=(1+0.75*dz.clip(-1.5,2.5)).clip(0,3)
F=(-r5/(rv*np.sqrt(5)+1e-12)).mul(mult,axis=0).shift(1)
rows=[]
for i in range(len(C)-10):
 f=F.iloc[i]; fr=C.iloc[i+1+9]/C.iloc[i+1]-1; x=pd.concat([f,fr],axis=1).dropna()
 if len(x)>=8: rows.append((C.index[i],spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic,len(x)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(q),'avgN',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4)); print('IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 z=q.loc[a:b]; print(a+'-'+b,'n',len(z),'ic',round(z.ic.mean(),6),'icir',round(z.ic.mean()/z.ic.std(ddof=1),6))
for h in [5,10,20]:
 rr=[]
 for i in range(len(C)-h):
  x=pd.concat([F.iloc[i],C.iloc[i+h]/C.iloc[i+1]-1],axis=1).dropna()
  if len(x)>=8: rr.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
 print('decay',h,len(rr),round(float(np.mean(rr)),6))
print('turnover',float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
F.to_csv('scripts/miner_1_20331209_dispersion_shock_reversal_signal.csv'); q.to_csv('scripts/miner_1_20331209_dispersion_shock_reversal_ic.csv')
