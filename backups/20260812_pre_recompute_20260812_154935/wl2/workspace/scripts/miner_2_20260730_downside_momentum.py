import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end]['close'] for s in U}; P=pd.concat(D,axis=1).sort_index(); r=P.pct_change()
ret=P/P.shift(20)-1; neg=r.clip(upper=0); down=np.sqrt((neg.pow(2)).rolling(60,min_periods=30).mean()); f=(ret/(down*np.sqrt(20))).shift(1); fr=P.pct_change().shift(-1)
def calc(y):
 rec=[]
 for d in f.index:
  x=f.loc[d]; z=y.loc[d]; ok=x.notna()&z.notna()
  if ok.sum()>=8: rec.append((d,x[ok].corr(z[ok]),ok.sum()))
 return pd.DataFrame(rec,columns=['date','ic','n'])
a=calc(fr); print('idea=20d return / 60d downside deviation, lag1'); print('dates',len(a),'avg_names',a.n.mean(),'coverage',a.n.mean()/15,'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',np.mean(a.ic>0))
for h in [5,10]:
 q=calc(P.pct_change(h).shift(-h)); print(h,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1))
for yr in range(2020,2027):
 q=a[a.date.dt.year==yr]
 if len(q): print(yr,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
# local provenance artifact
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20260730_downside_momentum_signals.csv',index=False)
