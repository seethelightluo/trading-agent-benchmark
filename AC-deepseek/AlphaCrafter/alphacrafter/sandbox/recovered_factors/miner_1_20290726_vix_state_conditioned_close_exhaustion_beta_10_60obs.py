import numpy as np, pandas as pd
from scipy.stats import spearmanr
# One idea: VIX-state-conditioned close-location exhaustion.  A close near its
# intraday high is reversed, with strength tilted toward assets whose trailing
# returns have the most negative response to VIX shocks during elevated VIX.
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2029-07-25'; ROOT='../persistent/stock_data'; IROOT='../persistent/index_data'
D={a:pd.read_csv(f'{ROOT}/{a}.csv').set_index('date').sort_index().loc[:END] for a in A}
v=pd.read_csv(f'{IROOT}/VIX.csv').set_index('date').sort_index().loc[:END]
ix=sorted(set().union(*[set(d.index) for d in D.values()]))
c=pd.DataFrame({a:D[a].reindex(ix).close for a in A}); hi=pd.DataFrame({a:D[a].reindex(ix).high for a in A}); lo=pd.DataFrame({a:D[a].reindex(ix).low for a in A}); vr=v.close.reindex(ix).ffill()
r=c.pct_change(); dv=vr.pct_change(); clv=((2*c-hi-lo)/(hi-lo).replace(0,np.nan)).clip(-1,1).rolling(10,min_periods=7).mean()
# rolling VIX beta, then cross-sectionally normalize: defensive VIX response is
# rewarded only in an objectively elevated VIX state.
beta=r.rolling(60,min_periods=40).cov(dv).div(dv.rolling(60,min_periods=40).var(),axis=0)
bz=beta.sub(beta.mean(axis=1),axis=0).div(beta.std(axis=1).replace(0,np.nan),axis=0).clip(-3,3)
vstate=((vr-vr.rolling(60,min_periods=40).mean())/vr.rolling(60,min_periods=40).std()).clip(-2,2)
# baseline exhaustion; high VIX additionally favors negative-VIX-beta assets.
f=-clv*(1-0.50*bz.mul(vstate.clip(lower=0),axis=0)); vis=c.index[c.index<=END]
def stat(sub,h):
 fw=c.shift(-h).div(c)-1; vals=[]; ns=[]; turns=[]; prev=None
 for t in sub:
  z=pd.concat([f.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  q=f.loc[t].rank(); zz=pd.concat([q,prev],axis=1).dropna() if prev is not None else pd.DataFrame()
  if len(zz)>=8: turns.append(1-spearmanr(zz.iloc[:,0],zz.iloc[:,1]).statistic)
  prev=q
 x=np.array(vals); return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns),np.mean(turns)
print('FACTOR vix_state_conditioned_close_exhaustion_beta_10_60obs cutoff',vis[-1],'assets',len(A),'cells',int(f.loc[vis].notna().sum().sum()),'of',len(vis)*15)
for h in [1,5,10,20]:
 x=stat(vis,h); print('H',h,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4),'mean_n',round(x[4],2),'coverage',round(f.loc[vis].notna().mean().mean(),4),'turn',round(x[5],4))
for lab,sub in [('2020_21',vis[vis<'2022-01-01']),('2022_23',vis[(vis>='2022-01-01')&(vis<'2024-01-01')]),('2024_25',vis[(vis>='2024-01-01')&(vis<'2026-01-01')]),('2026_current',vis[vis>='2026-01-01'])]:
 x=stat(sub,5); print('REGIME_5D',lab,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4))
