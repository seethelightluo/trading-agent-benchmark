import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2032-04-29')
def get(s):
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float); return x[x.index<=CUT].sort_index()
P=pd.concat({s:get(s) for s in U},axis=1).sort_index(); R=P.pct_change()
# Breadth-conditioned short reversal: reverse 3d move only when market breadth is extreme,
# with volatility scaling. Broad weakness selects reversal; broad strength selects continuation.
bread=(R.shift(1)>0).mean(axis=1); extreme=(bread<.30)|(bread>.70)
F=(-R.shift(1).rolling(3).sum()/R.shift(1).rolling(20).std()).where(extreme,0.0)
rows=[]; sig=[]
for i in range(25,len(P)-10):
 x=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+10]/P.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(x)>=8 and x.f.nunique()>1 and x.y.nunique()>1: rows.append((P.index[i],x.f.rank().corr(x.y.rank()),len(x)))
 sig.append(F.iloc[i].rename(P.index[i]))
I=pd.DataFrame(rows,columns=['date','ic','n']); mu=I.ic.mean(); sd=I.ic.std(ddof=1)
print({'dates':len(I),'avg_instruments':round(I.n.mean(),2),'coverage':round(len(I)/(len(P)-10),4),'ic_10d':round(mu,6),'icir_daily':round(mu/sd,6),'hit_ratio':round((I.ic>0).mean(),4),'period_start':str(I.date.min().date()),'period_end':str(I.date.max().date())})
for n in [60,180,365,756]:
 z=I.tail(n); print('recent',n,'ic',round(z.ic.mean(),6),'icir',round(z.ic.mean()/z.ic.std(ddof=1),6),'dates',len(z))
for h in [1,5,20]:
 a=[]
 for i in range(25,len(P)-h):
  x=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(x)>=8 and x.f.nunique()>1 and x.y.nunique()>1:a.append(x.f.rank().corr(x.y.rank()))
 print('decay',h,round(np.mean(a),6),len(a))
pd.DataFrame(sig).to_csv('scripts/miner_2_20320429_breadth_reversal_signal.csv'); I.to_csv('scripts/miner_2_20320429_breadth_reversal_ic.csv',index=False)
