import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}; P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# candidate: medium-term momentum penalized by long realized volatility
F=R.rolling(20).sum()/R.rolling(60).std()
def test(h):
 out=[]
 for i in range(60,len(P)-h):
  z=pd.concat([F.iloc[i],P.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(out); return len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
print('factor=20d return/60d vol; dates',sum(1 for i in range(60,len(P)-1) if F.iloc[i].notna().sum()>=8),'coverage',F.notna().sum().sum()/(F.shape[0]*F.shape[1]))
for h in [1,5,10,20]: print('h',h,test(h))
print('turnover10',np.nanmean((F.rank(axis=1,pct=True)-F.shift(10).rank(axis=1,pct=True)).abs().mean(axis=1)))
for lo,hi in [('2020','2024'),('2024','2028'),('2028','2031'),('2031','2033')]:
 vals=[]
 for i in range(60,len(P)-1):
  if not(str(P.index[i].date())>=lo+'-01-01' and str(P.index[i].date())<=hi+'-12-31'): continue
  z=pd.concat([F.iloc[i],R.iloc[i+1]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals);print('regime',lo,hi,len(a),a.mean() if len(a) else np.nan,(a.mean()/a.std(ddof=1)) if len(a)>1 else np.nan)
print('recent120',test(1)[1] if False else '')
vals=[]
for i in range(max(60,len(P)-120),len(P)-1):
 z=pd.concat([F.iloc[i],R.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
a=np.array(vals);print('recent120',len(a),a.mean(),a.mean()/a.std(ddof=1))
