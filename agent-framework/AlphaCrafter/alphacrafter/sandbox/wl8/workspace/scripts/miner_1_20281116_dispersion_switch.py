import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-11-14')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Cross-asset dispersion regime, lagged one day. In high dispersion, fade 5d idiosyncratic shocks;
# in low dispersion, use continuation. Signal is volatility-scaled lagged 5d return.
disp=r.rolling(20,min_periods=15).std().mean(axis=1).shift(1)
q=disp.rolling(120,min_periods=60).rank(pct=True)
reg=np.where(q>=.65,-1.,1.)
base=r.rolling(5,min_periods=5).sum().shift(1).div(r.rolling(20,min_periods=15).std().shift(1))
sig=base.mul(reg,axis=0).clip(-8,8)
for h in [1,3,5,10]:
 f=px.shift(-h)/px-1; rows=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); recent=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),recent.ic.mean(),recent.ic.mean()/recent.ic.std(ddof=1)))
print('artifact_dates',len(sig.index),'coverage',round(sig.notna().sum().sum()/sig.size,4),'nonzero',round((sig!=0).sum().sum()/sig.size,4))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20281116_dispersion_switch_signal.csv',index=False)
