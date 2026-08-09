import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date'); px[a]=d['close']
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Trend continuation is favored when recent volatility is compressed versus its own medium-term baseline.
# Signal: 20d momentum, scaled by positive compression (long vol / short vol), all lagged one session.
sv=r.rolling(5,min_periods=4).std(); lv=r.rolling(40,min_periods=25).std()
compression=(lv/sv).clip(0.25,4.0)
mom=p.pct_change(20,min_periods=15)
f=(mom*compression).shift(1)
print('candidate=volatility_compression_trend')
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; counts=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8:
   vals.append(spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic); counts.append(ok.sum())
 s=pd.Series(vals); print('h=%d dates=%d mean_valid=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(s),np.mean(counts),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rank=f.rank(axis=1,pct=True); print('coverage=%.6f turnover10=%.6f total_dates=%d assets=%d mean_valid=%.3f'%(f.notna().sum().sum()/f.size,(rank-rank.shift(10)).abs().mean(axis=1).mean(),len(f),len(assets),f.notna().sum(axis=1).mean()))
for start,end in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-09-19')]:
 fr=p.shift(-10)/p-1; vals=[]
 for dt in f.loc[start:end].index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic)
 s=pd.Series(vals); print('regime=%s/%s n=%d IC=%.6f ICIR=%.6f'%(start,end,len(s),s.mean(),s.mean()/s.std(ddof=1)))
