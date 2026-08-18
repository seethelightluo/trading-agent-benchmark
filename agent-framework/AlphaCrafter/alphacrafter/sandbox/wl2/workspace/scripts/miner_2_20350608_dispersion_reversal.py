import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for a in assets:
 d=pd.read_csv(os.path.join(base,a+'.csv'),parse_dates=['date']).set_index('date').sort_index(); px[a]=d['close'].replace(0,np.nan)
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change()
# Cross-asset dispersion: cross-sectional standard deviation of trailing 20d vol.
assetvol=ret.rolling(20).std(); csdisp=assetvol.std(axis=1)
disp_z=(csdisp-csdisp.rolling(120).mean())/csdisp.rolling(120).std()
raw=-ret.rolling(5).sum()/assetvol
sig=raw.where(disp_z>0.5)
for h in [5,10,20]:
 vals=[]
 for i in range(len(prices)-h-1):
  s=sig.iloc[i]; fr=prices.iloc[i+1:i+1+h].iloc[-1]/prices.iloc[i]-1
  z=pd.concat([s,fr],axis=1).dropna()
  if len(z)>=8: vals.append((prices.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z),(s.notna()).mean()))
 x=pd.DataFrame(vals,columns=['date','ic','n','coverage']); ic=x.ic
 print('H',h,'dates',len(x),'avgN',round(x.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(x.coverage.mean(),4))
 for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2035-05-24')]:
  q=x[(x.date>=lo)&(x.date<=hi)].ic
  if len(q): print(' ',lo,hi,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
 if h==10: x.to_csv('../persistent/miner_2_20350608_dispersion_reversal_signal.csv',index=False)
