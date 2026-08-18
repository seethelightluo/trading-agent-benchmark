import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; macro='../persistent/index_data/VIX.csv'
px={}
for s in U:
 f=os.path.join(base,s+'.csv')
 if os.path.exists(f): px[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').close
vix=pd.read_csv(macro,parse_dates=['date']).set_index('date').close.rename('vix')
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); vix=vix.reindex(P.index).ffill()
# Conditional macro sign-switch: medium momentum in calm, short reversal in stressed.
# signal at t uses data through t-1; compute raw then shift one day.
med=P.pct_change(40); short=P.pct_change(5); vol=R.rolling(20).std()
vz=(vix-vix.rolling(120).median())/vix.rolling(120).median()
raw=med.where(vz.lt(0), -short)
sig=(raw/vol).shift(1)
# forward non-overlapping daily origin observations, 10d forward
fwd=P.shift(-10)/P-1
rows=[]
for d in sig.index:
 x=sig.loc[d]; y=fwd.loc[d]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((d,ic,len(z),float(vz.loc[d]) if pd.notna(vz.loc[d]) else np.nan))
a=pd.DataFrame(rows,columns=['date','ic','n','vz']).set_index('date')
# same horizon daily paper IC, annualized? ICIR mean/std (as requested daily paper)
print('dates',len(a),'avgN',a.n.mean(),'coverage',a.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean(),sig.diff().abs().mean().mean()))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=a.loc[lo:hi].ic
 print(lo,hi,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
for h in [5,10,20,40]:
 y=P.shift(-h)/P-1; rr=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'IC',np.nanmean(rr),'n',len(rr))
# artifact
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_2_20340331_macro_switch_signal.csv',index=False)
