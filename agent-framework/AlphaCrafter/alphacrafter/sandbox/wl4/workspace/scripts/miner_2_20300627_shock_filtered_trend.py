import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data/'
frames={}
for s in U:
 d=pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 frames[s]=d['close'].astype(float)
px=pd.concat(frames,axis=1).sort_index()
px=px.loc[:'2030-06-26']
ret=px.pct_change()
# Shock-filtered trend: 20d risk-adjusted trend discounted when the latest move is abnormal.
trend=px.pct_change(20)/(ret.rolling(20).std()*np.sqrt(20))
shock=(ret.abs()/ret.rolling(60).std()).clip(upper=3)
f=trend*(1-shock/4)
rows=[]
for i in range(60,len(px)-10):
 x=f.iloc[i]; y=px.iloc[i+10]/px.iloc[i]-1
 ok=x.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((px.index[i],ok.sum(),spearmanr(x[ok],y[ok]).statistic, spearmanr(x[ok],y[ok]).pvalue))
r=pd.DataFrame(rows,columns=['date','n','ic','p']).set_index('date')
def stats(z): return (len(z),z.nunique(),z.mean(),z.std(ddof=1),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),(z>0).mean())
print('dates',len(r),'avg_n',r.n.mean(),'period',r.index.min().date(),r.index.max().date())
for name,z in [('full',r.ic),('recent252',r.ic.tail(252)),('recent120',r.ic.tail(120))]: print(name,'N mean std ICIR hit',stats(z))
print('coverage',r.n.mean()/15)
# rank turnover proxy based on daily signal ranks
rank=f.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna(); print('turnover_proxy',turn.loc[r.index].mean())
# decay horizons
for h in [1,5,10,20]:
 vals=[]
 for i in range(60,len(px)-h):
  x=f.iloc[i]; y=px.iloc[i+h]/px.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,len(vals),np.nanmean(vals))
