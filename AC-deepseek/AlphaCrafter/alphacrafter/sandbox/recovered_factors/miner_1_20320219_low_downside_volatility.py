import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index(); R=P.pct_change(); F=[]
for i,d in enumerate(P.index):
 if i<65:continue
 # defensive quality: low downside deviation over 60d, residualized by common median return
 e=R.iloc[i-60:i].sub(R.iloc[i-60:i].median(axis=1),axis=0)
 F.append((-e.where(e<0).std()).rename(d))
F=pd.DataFrame(F)
def go(h,ix):
 out=[]; ns=[]
 for d in ix:
  i=P.index.get_loc(d)
  if i+h>=len(P):continue
  z=pd.concat([F.loc[d],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(out);return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)
for h in [1,5,10,20]:print('H',h,go(h,F.index))
print('coverage',F.notna().mean().mean(),'dates',len(F),'assets',len(A),'turnover10',F.rank(pct=True).sub(F.rank(pct=True).shift(10)).abs().mean(axis=1).dropna().mean())
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 q=go(10,F.loc[lo:hi].index);print('REG',lo,hi,q[0],round(q[2],6),round(q[3],6))
q=go(10,F.index[-120:]);print('RECENT120',q[0],round(q[2],6),round(q[3],6))
