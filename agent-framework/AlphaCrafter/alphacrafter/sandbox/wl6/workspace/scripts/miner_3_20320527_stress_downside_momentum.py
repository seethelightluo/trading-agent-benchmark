import pandas as pd, numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-05-26')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index(); P=P[P.index<=CUT]; r=P.pct_change()
# Smooth downside risk and reward-to-risk momentum, activated only when macro stress is elevated.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill()
ret20=P/P.shift(20)-1
neg=r.clip(upper=0); down=np.sqrt((neg**2).rolling(60,min_periods=30).mean())*np.sqrt(20)
stress=((vix/vix.rolling(120,min_periods=60).median()).rank(pct=True)+ (dxy/dxy.rolling(120,min_periods=60).median()).rank(pct=True))/2
# stress is a common regime multiplier; downside-adjusted momentum retains cross-sectional ordering.
f=ret20.div(down).mul((0.5+stress).clip(0.5,1.5),axis=0)
def run(h):
 ic=[]; cov=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.x,z.y).statistic);cov.append(len(z)/15)
 x=np.array(ic); return len(x),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(cov)
print('cutoff',CUT.date(),'dates',len(P),'assets',len(A),'coverage',f.notna().stack().mean())
for h in [5,10,20]: print('horizon',h,run(h))
for yr in range(2026,2033):
 x=[]
 for i in range(len(P)-20):
  if P.index[i].year==yr:
   z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+20]/P.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(z)>=8:x.append(spearmanr(z.x,z.y).statistic)
 print('regime',yr,len(x),np.mean(x) if x else None)
# rank turnover
turn=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1].rank(pct=True).rename('a'),f.iloc[i].rank(pct=True).rename('b')],axis=1).dropna()
 if len(z):turn.append(np.mean(abs(z.b-z.a)))
print('turnover',np.mean(turn))
