import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2031-08-21')
P=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); P=P[P.index<=end]; r=P.pct_change()
# persistent trend quality: signed 20d return times directional efficiency, lagged
ret=P.shift(1)/P.shift(21)-1
eff=ret.abs()/(r.abs().rolling(20,min_periods=15).sum().shift(1)+1e-9)
F=ret*eff
Y=P.shift(-10)/P-1; rows=[]; turn=[]; prev=None
for d in F.index:
 z=pd.concat([F.loc[d],Y.loc[d]],axis=1).dropna()
 if len(z)>=8:
  rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  q=F.loc[d].rank(pct=True).dropna()
  if prev is not None:
   a=pd.concat([prev,q],axis=1).dropna();turn.append((a.iloc[:,0]-a.iloc[:,1]).abs().mean())
  prev=q
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(R),'avg_n',R.n.mean(),'coverage',R.n.mean()/15);print('IC',R.ic.mean(),'ICIR',R.ic.mean()/R.ic.std(ddof=1),'hit',(R.ic>0).mean(),'turnover',np.mean(turn))
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=R.loc[a:b].ic;print(a,b,len(q),q.mean() if len(q) else np.nan)
F.index=F.index.strftime('%Y-%m-%d');F.to_csv('scripts/miner_1_20310821_trend_quality20_signal.csv',index_label='date')
