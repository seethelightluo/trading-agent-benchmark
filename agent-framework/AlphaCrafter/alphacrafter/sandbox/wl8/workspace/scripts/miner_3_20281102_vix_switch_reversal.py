import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-10-31')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()
vc=(v['close'] if 'close' in v else v.iloc[:,0]).reindex(px.index).ffill()
# Lagged VIX percentile switch: momentum in calm regimes, reversal in stressed regimes.
# The regime multiplier is based only on completed observations and changes cross-sectional rank
pct=vc.rolling(60,min_periods=30).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).shift(1)
sign=np.where(pct>=0.60,-1.0,1.0)
base=px.pct_change(3).shift(1).div(r.rolling(20,min_periods=15).std().shift(1))
sig=base.mul(sign,axis=0).clip(-8,8)
for h in [1,3,5,10]:
 f=px.shift(-h)/px-1; rows=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('artifact_dates',len(sig.index),'coverage',round(sig.notna().sum().sum()/sig.size,4),'nonzero',round((sig!=0).sum().sum()/sig.size,4))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20281102_vix_switch_reversal_signal.csv',index=False)
