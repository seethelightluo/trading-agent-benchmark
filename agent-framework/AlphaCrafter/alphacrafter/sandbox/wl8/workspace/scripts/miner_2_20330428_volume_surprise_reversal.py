import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
px=pd.DataFrame({s:D[s].close.astype(float) for s in U}).loc[:'2033-04-27']
volu=pd.DataFrame({s:D[s].volume.astype(float).replace(0,np.nan) for s in U}).reindex(px.index)
r=px.pct_change(); rv=r.rolling(20).std()
# Volume-surprise weighted 10-day reversal: abnormal activity modulates the contrarian signal.
vs=(volu.rolling(5).mean()/volu.rolling(40).mean()).clip(0.25,4)
sig=-px.pct_change(10)/rv * np.sqrt(vs)
f=sig.rank(axis=1,pct=True)
def ic(a,b):
 n=(a.notna()&b.notna()).sum(axis=1); ar=a.rank(axis=1); br=b.rank(axis=1)
 am=ar.mean(axis=1); bm=br.mean(axis=1)
 aa=ar-am.values[:,None]; bb=br-bm.values[:,None]
 z=(aa*bb).sum(axis=1)/np.sqrt((aa*aa).sum(axis=1)*(bb*bb).sum(axis=1))
 return z.where(n>=8).dropna()
q=ic(f,px.shift(-10)/px-1)
print('candidate volume_surprise_invvol_reversal_10d dates',len(q),'avgN',((f.notna()&(px.shift(-10).notna())).sum(axis=1).loc[q.index]).mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean(),'turnover',f.diff().abs().sum(axis=1).div(2).loc[q.index].mean())
print('decay',[(h,ic(f,px.shift(-h)/px-1).mean()) for h in [1,5,10,20]])
# regime split
for label,ix in [('early',q.index<'2026-01-01'),('recent',q.index>='2030-01-01'),('last365',q.index>=q.index.max()-pd.Timedelta(days=365))]:
 z=q.loc[ix]; print(label,len(z),z.mean(),z.mean()/z.std() if len(z)>1 else np.nan)
print('coverage',f.notna().mean().mean())
# recoverable artifact
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20330428_volume_surprise_invvol_reversal_10d_signal.csv',index=False)
q.rename('ic').to_csv('scripts/miner_2_20330428_volume_surprise_invvol_reversal_10d_ic.csv')
