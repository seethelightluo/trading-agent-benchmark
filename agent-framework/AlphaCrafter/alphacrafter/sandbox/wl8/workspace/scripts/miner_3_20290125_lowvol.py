import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-01-23')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Defensive low-volatility: inverse lagged 20d realized volatility, with a mild
# 60d volatility anchor to avoid one-day noise. Every input is lagged one day.
v20=r.rolling(20,min_periods=15).std().shift(1)
v60=r.rolling(60,min_periods=40).std().shift(1)
sig=(1/(v20*(v60**0.25))).replace([np.inf,-np.inf],np.nan)
sig=sig.rank(axis=1,pct=True)
for h in [1,3,5,10]:
 f=px.shift(-h)/px-1; rows=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:
   rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN %.2f'%z.n.mean(),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('coverage',sig.notna().sum().sum()/sig.size,'nonzero',(sig!=0).sum().sum()/sig.size)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20290125_lowvol_signal.csv',index=False)
