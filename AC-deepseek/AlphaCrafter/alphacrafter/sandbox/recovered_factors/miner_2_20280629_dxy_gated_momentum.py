import pandas as pd,numpy as np,glob,os,json
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}
px=pd.concat({a:D[a].close for a in A},axis=1).sort_index(); R=px.pct_change()
# Candidate: medium-term momentum gated by the *lagged* direction of the observation-only DXY trend.
# A rising dollar can alter cross-asset continuation; multiplying by signed DXY 20d trend tests conditional rather than raw momentum.
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(px.index).ffill()
dxy_sig=np.sign(dxy.pct_change(20)).replace(0,np.nan)
f=(R.rolling(20,min_periods=15).sum()*dxy_sig).shift(1)
print('assets',len(A),'dates',px.index.min().date(),px.index.max().date(),'candidate lagged DXY-gated 20d momentum')
def calc(h, mask=None):
 y=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  if mask is not None and not mask.loc[dt]: continue
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 a=np.asarray(vals); return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()
for h in [1,5,10,20]: print('H',h,'dates %d avgN %.2f IC %.6f ICIR %.6f hit %.4f'%calc(h))
for y in range(2020,2029):
 o=calc(1,pd.Series(f.index.year==y,index=f.index)); print('YEAR',y,'dates %d avgN %.2f IC %.6f ICIR %.6f hit %.4f'%o)
print('coverage %.4f avg_valid %.2f turnover %.5f'%(f.notna().sum().sum()/f.size,f.notna().sum(axis=1).mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
# Library correlation evidence: reconstruct admitted signal proxies from their documented formulas.
lib={
'risk_trend':(R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std()).shift(1),
'ravmom':(R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std()).shift(1),
'volnorm_rev5':(-R.rolling(5,min_periods=4).sum()/R.rolling(5,min_periods=4).std()).shift(1),
'volshock':(R.rolling(5,min_periods=5).sum()*(R.rolling(5,min_periods=5).std()/R.rolling(40,min_periods=20).std()-1)).shift(1),
'vol_transition':(R.rolling(5,min_periods=5).std()/R.rolling(60,min_periods=30).std()).shift(1),
'range_accel':(((px-px.rolling(20).min())/(px.rolling(20).max()-px.rolling(20).min())-.5)-((px-px.rolling(60).min())/(px.rolling(60).max()-px.rolling(60).min())-.5)).shift(1),
'peer_lead':((R.rolling(2).sum().mean(axis=1).values[:,None]-R.rolling(2).sum()).astype(float)),
}
mx=0; pair=''
for n,x in lib.items():
 z=pd.concat([f.stack(),x.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 rho=abs(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic) if len(z)>20 else np.nan
 print('CORR',n,'%.6f'%rho,'n',len(z))
 if np.isfinite(rho) and rho>mx:mx=rho;pair=n
print('MAX_LIBRARY_CORRELATION %.6f %s'%(mx,pair))
