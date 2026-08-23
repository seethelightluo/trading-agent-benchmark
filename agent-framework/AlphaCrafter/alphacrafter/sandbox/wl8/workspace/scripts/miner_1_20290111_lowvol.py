import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-01-09')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); P[s]=d.loc[d.index<=END,'close']
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); v=r.rolling(20,min_periods=15).std().shift(1)
# Defensive low-volatility signal, with a mild drawdown penalty; all inputs lagged.
dd=px.div(px.rolling(60,min_periods=40).max().shift(1))-1
sig=(-v - 0.25*dd.abs()).clip(-8,8)
for h in [1,3,5,10]:
 f=px.shift(-h)/px-1; out=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: out.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); q=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4),'recentIC',round(q.ic.mean(),6),'recentICIR',round(q.ic.mean()/q.ic.std(ddof=1),6))
print('coverage',sig.notna().sum().sum()/sig.size)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20290111_lowvol_signal.csv',index=False)
