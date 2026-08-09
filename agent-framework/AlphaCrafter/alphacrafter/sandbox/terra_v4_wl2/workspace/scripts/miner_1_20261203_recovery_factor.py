import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Candidate: recovery/oversold factor, distance above trailing 60d low, with recent 5d reversal
# test pure recovery and blended recovery-reversal configurations
configs={'recovery60':P/P.rolling(60,min_periods=45).min()-1,
         'recovery60_rev5':(P/P.rolling(60,min_periods=45).min()-1)-R.rolling(5,min_periods=5).sum(),
         'recovery120':P/P.rolling(120,min_periods=80).min()-1}
for name,F in configs.items():
 for h in [1,5,10]:
  vals=[]; dates=[]
  for i in range(len(P)-h):
   f=F.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1
   z=pd.concat([f,y],axis=1).dropna()
   if len(z)>=8:
    vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(P.index[i])
  a=np.array(vals); print(name,h,'dates',len(a),'names',round(np.mean([len(pd.concat([F.loc[d],(P.shift(-h).loc[d]/P.loc[d]-1)],axis=1).dropna()) for d in dates]),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4))
 # regime split and coverage/turnover
 print('coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round((F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),4))
