import pandas as pd, numpy as np
from scipy.stats import spearmanr
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2034-03-15'
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].replace(0,np.nan) for s in watch}).sort_index().loc[:END]
r=px.pct_change(); common=r.median(axis=1)
beta=r.rolling(90,min_periods=70).cov(common).div(common.rolling(90,min_periods=70).var(),axis=0)
res=r-beta.mul(common,axis=0)
# Broad, interpretable idiosyncratic range-compression: lower recent residual volatility relative to own 60d baseline scores higher.
short=res.rolling(10,min_periods=8).std(); long=res.rolling(60,min_periods=45).std()
sig=-(short/long.replace(0,np.nan)).clip(upper=5)
def stats(a):
 return (round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)) if len(a)>1 and a.std(ddof=1)>0 else (np.nan,np.nan,np.nan)
def report(h):
 vals=[]; counts=[]; fwd=px.shift(-h).div(px)-1
 for d in sig.index:
  ok=sig.loc[d].notna()&fwd.loc[d].notna()
  if ok.sum()>=8:
   v=spearmanr(sig.loc[d,ok],fwd.loc[d,ok]).statistic
   if np.isfinite(v): vals.append((d,v));counts.append(ok.sum())
 a=np.array([v for _,v in vals]); print('H',h,'dates',len(a),'mean_names',round(np.mean(counts),3),'IC/ICIR/hit',stats(a))
 for n,l,u in [('2020_2027','2020-01-01','2028-01-01'),('2028_2030','2028-01-01','2031-01-01'),('2031_now','2031-01-01','2100-01-01'),('latest_6m','2033-09-15','2100-01-01')]:
  q=np.array([v for d,v in vals if pd.Timestamp(l)<=d<pd.Timestamp(u)]); print(' ',n,'n',len(q),'IC/ICIR/hit',stats(q))
for h in (1,5,10,20): report(h)
print('coverage',round(sig.notna().mean().mean(),6),'cells',int(sig.notna().sum().sum()),'of',sig.size)
ranks=sig.rank(axis=1,pct=True); print('rank_turnover',round(ranks.diff().abs().stack().mean(),6),'median_iqr',round(sig.quantile(.75,axis=1).sub(sig.quantile(.25,axis=1)).median(),6))
print('endpoint',px.index.max().date(),'rows',len(px),'assets',px.shape[1])
