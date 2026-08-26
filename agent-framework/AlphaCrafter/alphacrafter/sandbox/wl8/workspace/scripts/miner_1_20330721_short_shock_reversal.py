import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
wide=pd.DataFrame(px).sort_index().loc[:'2033-07-20']
r=np.log(wide).diff(); vol=r.rolling(20,min_periods=15).std()
# short shock reversal, volatility normalized and smoothed, higher score means expected positive forward return
f=-(r.rolling(3).sum()/vol.shift(1)).rolling(3).mean()
rows=[]
for i in range(25,len(wide)-10):
 d=wide.index[i]; y=wide.iloc[i+1:i+11].iloc[-1]/wide.iloc[i]-1
 x=f.iloc[i]
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((d,len(z),spearmanr(z.x,z.y).statistic))
df=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(df),'avgN',df.n.mean(),'coverage',df.n.sum()/(len(df)*15))
for h in [1,5,10,20]:
 vals=[]
 for i in range(25,len(wide)-h):
  x=f.iloc[i]; y=wide.iloc[i+h]/wide.iloc[i]-1; z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.x,z.y).statistic)
 a=np.array(vals); print('h',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'hit',(a>0).mean(),'n',len(a))
for label,days in [('full',None),('recent365',365),('recent750',750)]:
 a=df.ic.iloc[-days:] if days else df.ic
 print(label,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'hit',(a>0).mean())
# turnover proxy rank changes
ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean().mean())
df.to_csv('scripts/miner_1_20330721_short_shock_reversal_ic.csv')
