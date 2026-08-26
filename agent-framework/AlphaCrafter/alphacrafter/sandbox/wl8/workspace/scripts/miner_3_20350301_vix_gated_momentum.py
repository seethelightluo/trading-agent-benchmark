import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-03-01')
def load(p):
 d=pd.read_csv(p,parse_dates=['date']).set_index('date'); return d['close'].astype(float)
px=pd.concat({a:load('../persistent/stock_data/'+a+'.csv') for a in assets},axis=1).sort_index().loc[:END].ffill()
vix=load('../persistent/index_data/VIX.csv').reindex(px.index).ffill()
r=px.pct_change(); mom=px.pct_change(20); vol=r.rolling(30).std()*np.sqrt(252)
vixmed=vix.rolling(252,min_periods=80).median()
state=(vix.shift(1)<=vixmed.shift(1)).astype(float)*0.65+0.35
sig=(mom/vol).shift(1).mul(state,axis=0)
fwd=px.shift(-10)/px-1
rows=[]
for dt in px.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z): rows.append((dt,z,int(ok.sum())))
ic=np.array([z[1] for z in rows]); mean=ic.mean(); sd=ic.std(ddof=1); icir=mean/sd*np.sqrt(252)
rank=sig.rank(axis=1,pct=True); tr=rank.diff().abs().mean(axis=1).reindex([z[0] for z in rows]).mean()
print('period',rows[0][0].date(),rows[-1][0].date(),'dates',len(rows),'avgN',np.mean([z[2] for z in rows]))
print('IC10',mean,'ICIR_daily',icir,'hit',np.mean(ic>0),'turnover',tr,'coverage',np.mean([z[2]/15 for z in rows]))
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; out=[]
 for dt in px.index:
  x=sig.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: out.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.nanmean(out),len(out))
for n in [365,750,1260]:
 q=ic[-min(n,len(ic)):]; print('recent',n,'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'IC',q.mean(),'dates',len(q))
out=pd.DataFrame({'date':sig.index});
for a in assets: out[a]=sig[a].values
out.to_csv('scripts/miner_3_20350301_vix_gated_momentum_signal.csv',index=False)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20350301_vix_gated_momentum_ic.csv',index=False)
