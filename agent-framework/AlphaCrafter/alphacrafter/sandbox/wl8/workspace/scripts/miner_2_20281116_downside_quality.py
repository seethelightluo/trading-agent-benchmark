import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-11-15')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.sort_index()
 P[s]=x[x.index<=END]
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Downside-quality: reward positive-return asymmetry and penalize downside volatility.
# All rolling statistics are shifted one session, so the signal is known at decision time.
down=r.where(r<0,0.0).pow(2).rolling(30,min_periods=20).mean().pow(.5)
total=r.rolling(30,min_periods=20).std()
up=r.where(r>0,0.0).pow(2).rolling(30,min_periods=20).mean().pow(.5)
# asymmetry, stabilized by total volatility; bounded for cross-sectional robustness
sig=((up-down).div(total.replace(0,np.nan))).shift(1).clip(-5,5)
for h in [1,3,5,10,15]:
 f=px.shift(-h)/px-1; rows=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:
   rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('artifact_dates',len(sig.index),'coverage',round(sig.notna().sum().sum()/sig.size,4),'nonzero',round((sig!=0).sum().sum()/sig.size,4))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20281116_downside_quality_signal.csv',index=False)
