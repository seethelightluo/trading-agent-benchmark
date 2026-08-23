import os, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-04-15')
def fetch(s):
 p='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].astype(float); return x[x.index<=CUT].sort_index()
P=pd.concat({s:fetch(s) for s in U},axis=1).sort_index(); R=P.pct_change()
# Elevated cross-asset dispersion gates a volatility-scaled 3-day reversal.
disp=R.rolling(20).std().mean(axis=1); gate=disp>disp.rolling(252,min_periods=100).quantile(.70)
F=-R.rolling(3).sum()/R.rolling(20).std(); F=F.where(gate,0.0)
rows=[]; sig=[]
for i in range(20,len(P)-10):
 f=F.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; x=pd.concat([f.rename('f'),y.rename('y')],axis=1).dropna()
 if len(x)>=8 and x.f.nunique()>1 and x.y.nunique()>1: rows.append((P.index[i],x.f.rank().corr(x.y.rank()),len(x)))
 sig.append(F.iloc[i].rename(P.index[i]))
I=pd.DataFrame(rows,columns=['date','ic','n']); mu=I.ic.mean(); sd=I.ic.std(ddof=1)
print({'dates':len(I),'avg_instruments':I.n.mean(),'coverage':len(I)/(len(P)-10),'ic_10d':mu,'icir_daily':mu/sd,'hit_ratio':(I.ic>0).mean(),'period_start':str(I.date.min()),'period_end':str(I.date.max())})
for n in [60,180,365]:
 z=I.tail(n); print('recent',n,'ic',z.ic.mean(),'icir',z.ic.mean()/z.ic.std(ddof=1),'dates',len(z))
pd.DataFrame(sig).to_csv('scripts/miner_2_20320415_dispersion_shock_reversal_signal.csv'); I.to_csv('scripts/miner_2_20320415_dispersion_shock_reversal_ic.csv',index=False)
