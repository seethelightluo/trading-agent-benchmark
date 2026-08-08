import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close']
 px[a]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# interpretable curvature: recent 20d risk-adjusted return minus slower 60d risk-adjusted return
m20=r.rolling(20,min_periods=15).sum(); m60=r.rolling(60,min_periods=40).sum()
v20=r.rolling(20,min_periods=15).std()*np.sqrt(20); v60=r.rolling(60,min_periods=40).std()*np.sqrt(60)
f=(m20/v20)-(m60/v60)
# lag signal one complete day
f=f.shift(1)
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h)
 vals=[]; n=[]; turns=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(x[ok],y[ok]).statistic); n.append(ok.sum())
   turns.append((x[ok].rank(pct=True)-f.shift(10).loc[dt][ok].rank(pct=True)).abs().mean() if dt in f.shift(10).index else np.nan)
 ic=np.array(vals); print('H',h,'dates',len(ic),'meanN',round(np.mean(n),2),'IC',round(np.nanmean(ic),5),'ICIR',round(np.nanmean(ic)/np.nanstd(ic,ddof=1),5),'hit',round(np.mean(ic>0),4),'turn10',round(np.nanmean(turns),4))
print('coverage',round(f.notna().stack().mean(),4),'period',p.index.min().date(),p.index.max().date())
# regime means for h10
fr=p.pct_change(10).shift(-10); z=[]
for dt in f.index:
 ok=f.loc[dt].notna()&fr.loc[dt].notna()
 if ok.sum()>=8:z.append((dt,spearmanr(f.loc[dt][ok],fr.loc[dt][ok]).statistic))
z=pd.Series(dict(z)); print('regimes',[(y,round(z[z.index.year==y].mean(),5),len(z[z.index.year==y])) for y in sorted(z.index.year.unique())])
