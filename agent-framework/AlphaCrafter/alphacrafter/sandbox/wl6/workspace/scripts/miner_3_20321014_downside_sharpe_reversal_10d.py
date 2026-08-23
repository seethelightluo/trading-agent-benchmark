import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index(); P=P[P.index<=pd.Timestamp('2032-10-13')]; r=P.pct_change()
ret10=P/P.shift(10)-1
neg=r.clip(upper=0); down=np.sqrt((neg**2).rolling(20,min_periods=15).mean())*np.sqrt(10)
# reversal of recent return, scaled by downside risk; high score = weak return with low downside risk
f=-ret10.div(down.replace(0,np.nan))
def run(h):
 ic=[]; cov=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.x,z.y).statistic);cov.append(len(z)/15)
 x=np.array(ic);return len(x),float(np.mean(x)),float(np.mean(x)/np.std(x,ddof=1)*np.sqrt(252)),float(np.mean(x>0)),float(np.mean(cov))
print('dates',len(P),'assets',len(A),'coverage',float(f.notna().stack().mean()))
for h in [5,10,20,40]:print('horizon',h,run(h))
for yr in range(2020,2033):
 x=[]
 for i in range(len(P)-10):
  if P.index[i].year==yr:
   z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+10]/P.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(z)>=8:x.append(spearmanr(z.x,z.y).statistic)
 print('regime',yr,len(x),float(np.mean(x)) if x else None)
turn=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1].rank(pct=True).rename('a'),f.iloc[i].rank(pct=True).rename('b')],axis=1).dropna()
 if len(z):turn.append(np.mean(abs(z.b-z.a)))
print('turnover',float(np.mean(turn)))
