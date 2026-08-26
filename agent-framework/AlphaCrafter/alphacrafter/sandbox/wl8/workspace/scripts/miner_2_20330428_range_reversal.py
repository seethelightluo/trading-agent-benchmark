import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).loc[:'2033-04-27']
r=px.pct_change(); vol=r.rolling(20).std(); rank=lambda x:x.rank(axis=1,pct=True)
def ic(a,b):
 n=(a.notna()&b.notna()).sum(axis=1); ar=a.rank(axis=1);br=b.rank(axis=1); aa=ar-ar.mean(axis=1).values[:,None];bb=br-br.mean(axis=1).values[:,None];z=(aa*bb).sum(axis=1)/np.sqrt((aa*aa).sum(axis=1)*(bb*bb).sum(axis=1));return z.where(n>=8).dropna()
# Test a single interpretable idea: reversal amplified for assets far from their 60d range midpoint.
pos=(px-px.rolling(60).min())/(px.rolling(60).max()-px.rolling(60).min())
base=-px.pct_change(10)/vol
variants={'range_extreme_invvol_rev10':base*(1+0.8*(2*pos.sub(.5).abs())),'range_midpoint_invvol_rev10':base*(1-0.8*(2*pos.sub(.5).abs()))}
for k,sig in variants.items():
 f=rank(sig);q=ic(f,px.shift(-10)/px-1); print(k,'dates',len(q),'avgN',((f.notna()&px.shift(-10).notna()).sum(axis=1).loc[q.index]).mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean(),'turn',f.diff().abs().sum(axis=1).div(2).loc[q.index].mean(),'cov',f.notna().mean().mean());print('decay',[(h,ic(f,px.shift(-h)/px-1).mean()) for h in [1,5,10,20]]); 
 for label,ix in [('early',q.index<'2026-01-01'),('recent',q.index>='2030-01-01'),('last365',q.index>=q.index.max()-pd.Timedelta(days=365))]:
  z=q.loc[ix];print(label,len(z),z.mean(),z.mean()/z.std())
 if k=='range_extreme_invvol_rev10':
  f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20330428_range_extreme_invvol_reversal_10d_signal.csv',index=False)
