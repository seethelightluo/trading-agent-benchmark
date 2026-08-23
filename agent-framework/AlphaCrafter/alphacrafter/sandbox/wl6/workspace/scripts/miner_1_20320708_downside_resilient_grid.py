import pandas as pd,numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-07-07'); A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index().loc[:CUT]; r=P.pct_change(); neg=r.clip(upper=0)
for look,vol in [(30,20),(40,30),(60,40),(60,20)]:
 dv=np.sqrt((neg**2).rolling(vol,min_periods=max(10,vol//2)).mean()); f=(P/P.shift(look)-1).div(dv)
 out=[]
 for i in range(len(P)-10):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+10]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.x,z.y).statistic)
 x=np.array(out); print('variant',look,vol,'dates',len(x),'avg_n',round(f.notna().sum(axis=1).mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
