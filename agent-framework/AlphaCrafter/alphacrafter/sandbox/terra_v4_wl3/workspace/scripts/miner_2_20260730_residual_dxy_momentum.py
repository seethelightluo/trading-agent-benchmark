import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close']
R=pd.DataFrame({s:x.pct_change() for s,x in px.items()}); dr=dxy.pct_change()
# residual momentum: 20d return net of rolling 60d DXY beta times DXY 20d return
betas=R.rolling(60,min_periods=45).cov(dr).div(dr.rolling(60,min_periods=45).var(),axis=0)
ret20=R.rolling(20,min_periods=20).sum(); d20=dr.rolling(20,min_periods=20).sum()
F=ret20-betas.mul(d20,axis=0)
# forward one-day return, aligned and date by date
ics=[]; turns=[]; validdates=0; nsum=0
prev=None
for i in range(len(F)-1):
 d=F.index[i]; nxt=F.index[i+1]; a=F.loc[d]; y=R.loc[nxt]
 z=pd.concat([a,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); validdates+=1; nsum+=len(z)
  ranks=a.rank(pct=True); turns.append(np.nan if prev is None else np.mean((ranks-prev).abs())) ; prev=ranks
ics=np.array(ics); turns=np.array(turns,dtype=float)
print('dates',validdates,'avg_names',nsum/validdates,'coverage',nsum/(validdates*15))
print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(np.nanmean(ics),np.nanmean(ics)/np.nanstd(ics,ddof=1),np.mean(ics>0),np.nanmean(turns)))
for h in [5,10]:
 vals=[]
 for i in range(len(F)-h):
  z=pd.concat([F.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 v=np.array(vals); print(h,'IC %.6f ICIR %.6f n'% (v.mean(),v.mean()/v.std(ddof=1)),len(v))
for yr in range(2020,2027):
 q=[x for i,x in enumerate(ics) if F.index[i].year==yr]
 if q: print(yr,'n',len(q),'mean',np.mean(q),'icir',np.mean(q)/np.std(q,ddof=1))
# pooled correlation to existing signal artifacts unavailable; report NA
