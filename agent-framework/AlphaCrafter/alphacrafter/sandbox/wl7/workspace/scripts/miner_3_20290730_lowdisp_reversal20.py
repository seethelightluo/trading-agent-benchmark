import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-07-30'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=P.pct_change(); lag=R.shift(1)
# contrarian 20d trend, active only in low cross-asset dispersion; all inputs lagged
raw=-(P.shift(1)/P.shift(21)-1)/R.shift(1).rolling(20,min_periods=15).std()
disp=lag.std(axis=1).where(lag.notna().sum(axis=1)>=8)
threshold=disp.rolling(252,min_periods=126).median().shift(1)
for name,gate in [('low',(disp<=threshold)),('mid',(disp>threshold)&(disp<=disp.rolling(252,min_periods=126).quantile(.75).shift(1))),('high',(disp>disp.rolling(252,min_periods=126).quantile(.75).shift(1)))]:
 sig=raw.mul(gate.astype(float),axis=0).replace([np.inf,-np.inf],np.nan)
 print('\nSTATE',name,'active',round(gate.mean(),4))
 for h in [10,20,40]:
  f=P.shift(-h)/P-1; out=[]
  for dt in P.index:
   z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).dropna()
   if len(z)>=8 and z.x.nunique()>1: out.append(spearmanr(z.x,z.y).statistic)
  q=pd.Series(out).dropna(); print(h,'dates',len(q),'avg_n',round(sig.notna().sum(axis=1).loc[sig.notna().sum(axis=1)>=8].mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
  if h==20 and name=='low': sig.to_csv('scripts/miner_3_20290730_lowdisp_reversal20_signal.csv')
