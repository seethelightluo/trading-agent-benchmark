import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-04-26')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
d={a:load(a) for a in assets}; px=pd.concat({a:d[a]['close'].astype(float) for a in assets},axis=1).sort_index().loc[:END].ffill(); r=px.pct_change()
# Five-day average close-location value, inverted so weak closes are favored; scale by 20d vol.
high=pd.concat({a:d[a]['high'].astype(float) for a in assets},axis=1).reindex(px.index).ffill(); low=pd.concat({a:d[a]['low'].astype(float) for a in assets},axis=1).reindex(px.index).ffill()
clv=((2*px-low-high)/(high-low).replace(0,np.nan)).rolling(5).mean()
vol=r.rolling(20).std()
sig=(-clv/(vol+0.01)).shift(1)
rows=[]
for dt in px.index:
 x=sig.loc[dt]; y=px.shift(-10).loc[dt]/px.loc[dt]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z): rows.append((dt,z,int(ok.sum())))
ic=np.array([z for _,z,_ in rows]); mean=ic.mean(); sd=ic.std(ddof=1)
print('period',rows[0][0].date(),rows[-1][0].date(),'dates',len(rows),'avgN',np.mean([n for _,_,n in rows]))
print('IC10',mean,'ICIR_daily',mean/sd*np.sqrt(252),'hit',np.mean(ic>0),'coverage',np.mean([n/15 for _,_,n in rows]))
rank=sig.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).reindex([x[0] for x in rows]).mean())
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; out=[]
 for dt in px.index:
  x=sig.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: out.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.nanmean(out),'dates',len(out))
for n in [365,750,1260]:
 q=ic[-min(n,len(ic)):]; print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'dates',len(q))
out=pd.DataFrame({'date':sig.index});
for a in assets: out[a]=sig[a].values
out.to_csv('factors/miner_1_20350426_clv_volscaled_reversal_10d_signal.csv',index=False)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('factors/miner_1_20350426_clv_volscaled_reversal_10d_ic.csv',index=False)
