import pandas as pd,numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-05-26'); A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index();P=P[P.index<=CUT]; r=P.pct_change()
# Directional efficiency: net 40d move divided by path length, multiplied by risk-adjusted 40d trend.
net=P/P.shift(40)-1; path=r.abs().rolling(40,min_periods=25).sum(); eff=net.div(path)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(40); f=eff*net.div(vol)
def run(h):
 x=[];c=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.x,z.y).statistic);c.append(len(z)/15)
 x=np.array(x);return len(x),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(c)
print('cutoff',CUT.date(),'dates',len(P),'assets',len(A),'coverage',f.notna().stack().mean())
for h in [5,10,20]:print('horizon',h,run(h))
for yr in range(2026,2033):
 x=[]
 for i in range(len(P)-20):
  if P.index[i].year==yr:
   z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+20]/P.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(z)>=8:x.append(spearmanr(z.x,z.y).statistic)
 print('regime',yr,len(x),np.mean(x) if x else None)
turn=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1].rank(pct=True).rename('a'),f.iloc[i].rank(pct=True).rename('b')],axis=1).dropna()
 if len(z):turn.append(np.mean(abs(z.b-z.a)))
print('turnover',np.mean(turn))