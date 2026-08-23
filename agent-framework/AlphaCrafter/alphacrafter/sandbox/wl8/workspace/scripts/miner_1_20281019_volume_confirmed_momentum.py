import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-10-17')
P={}; V={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 x=x[x.date<=END].set_index('date'); P[s]=x.close; V[s]=x.volume
px=pd.DataFrame(P).sort_index(); vol=pd.DataFrame(V).reindex(px.index)
r=px.pct_change();
# Volume-confirmed medium momentum: lagged 10d return, amplified by abnormal turnover, risk normalized.
ret10=px.pct_change(10).shift(1); rv=r.rolling(20,min_periods=15).std().shift(1)
vr=(vol/vol.rolling(20,min_periods=15).mean()).shift(1).clip(0.5,2.0)
sig=(ret10/rv * (0.5+0.5*vr)).clip(-8,8)
for h in [1,3,5,10]:
 f=px.shift(-h)/px-1; rows=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('artifact_dates',len(sig.index),'coverage',round(sig.notna().sum().sum()/sig.size,4),'nonzero',round((sig!=0).sum().sum()/sig.size,4))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20281019_volume_confirmed_momentum_signal.csv',index=False)
