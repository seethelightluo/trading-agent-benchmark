import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in syms:
 f='../persistent/stock_data/'+s+'.csv'; d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close']; P[s]=d
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# leadership: 20d return relative to contemporaneous cross-asset median, interpretable breadth factor
raw=px.pct_change(20); fac=raw.sub(raw.median(axis=1),axis=0)
# alternative: residual 20d return after beta to equal-weight market over rolling 60d
m=r.mean(axis=1); beta=r.rolling(60).cov(m).div(m.rolling(60).var(),axis=0); resid=raw-beta*m.rolling(20).sum().values[:,None] if False else None
for name,f in [('relative_leadership20',fac),('relative_leadership60',px.pct_change(60).sub(px.pct_change(60).median(axis=1),axis=0))]:
 print('\n',name)
 for h in [1,5,10]:
  y=px.pct_change(h).shift(-h)
  vals=[]; dates=[]; turnovers=[]
  for dt in f.index:
   a=f.loc[dt]; b=y.loc[dt]; ok=a.notna()&b.notna()
   if ok.sum()>=8:
    vals.append(spearmanr(a[ok],b[ok]).statistic); dates.append(dt)
  z=pd.Series(vals,index=dates).dropna()
  print('h',h,'dates',len(z),'assets_avg',round(np.mean([((f.loc[d].notna()&y.loc[d].notna()).sum()) for d in z.index]),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
 # rank turnover
 ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean().mean(),'coverage',f.notna().sum(axis=1).mean()/15)
# raw momentum comparison correlations pooled
print('corr leadership20 with raw20',fac.stack().corr(raw.stack()))
# macro conditional: leadership only when VIX 5d change positive? compute daily 5d forward
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(px.index).ffill(); cond=v.pct_change(5)>0
for label,mask in [('vix_up',cond),('vix_down',~cond)]:
 f=fac.where(mask, np.nan); y=px.pct_change(5).shift(-5); z=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic)
 z=pd.Series(z).dropna();print(label,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
