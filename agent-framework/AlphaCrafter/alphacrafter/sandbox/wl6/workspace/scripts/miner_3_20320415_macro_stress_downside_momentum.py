import pandas as pd, numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-04-14'); assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index();P=P[P.index<=CUT];r=P.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill()
ret20=P/P.shift(20)-1; neg=r.clip(upper=0); down=np.sqrt((neg**2).rolling(40,min_periods=10).mean())*np.sqrt(20)
stress=(vix/vix.rolling(60,min_periods=40).median()).clip(0.5,2.0);f=ret20.div(down).mul(stress,axis=0)
def calc(h):
 vals=[];cov=[];turn=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:vals.append(spearmanr(z.x,z.y).statistic);cov.append(len(z)/15)
 for i in range(1,len(f)):
  z=pd.concat([f.iloc[i-1].rename('a'),f.iloc[i].rename('b')],axis=1).dropna()
  if len(z):turn.append(np.mean(abs(z.b.rank(pct=True)-z.a.rank(pct=True))))
 a=np.array(vals);return len(a),round(np.mean(a),6),round(np.mean(a)/np.std(a,ddof=1),6),round(np.mean(a>0),4),round(np.mean(cov),4),round(np.mean(turn),4)
print('cutoff',CUT.date(),'dates',len(P),'assets',P.shape[1],'coverage',round(f.notna().stack().mean(),4))
for h in [5,10,20]:print(h,calc(h))
for yr in range(2026,2033):
 a=[]
 for i in range(len(P)-20):
  if P.index[i].year==yr:
   z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+20]/P.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.x,z.y).statistic)
 print('regime',yr,len(a),round(np.mean(a),6) if a else None)
