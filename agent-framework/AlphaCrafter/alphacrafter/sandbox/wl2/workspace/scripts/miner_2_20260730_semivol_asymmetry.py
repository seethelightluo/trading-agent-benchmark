import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end]['close'] for s in U}
P=pd.concat(D,axis=1).sort_index(); r=P.pct_change()
# Positive/negative semivolatility asymmetry: assets with relatively more upside than downside risk
up=np.sqrt(r.clip(lower=0).pow(2).rolling(20,min_periods=12).mean())
dn=np.sqrt(r.clip(upper=0).pow(2).rolling(20,min_periods=12).mean())
F=(up/(dn+1e-8)-1).shift(1); print('idea=20d upside/downside semivolatility asymmetry, lag1')
def calc(Y):
 out=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8: out.append((dt,z.f.corr(z.y),len(z)))
 return pd.DataFrame(out,columns=['date','ic','n'])
for h in [1,5,10]:
 a=calc(P.pct_change(h).shift(-h)); print('horizon',h,'dates',len(a),'avg_names',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   q=a[a.date.dt.year==yr]
   if len(q): print('regime',yr,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20260730_semivol_asymmetry_signals.csv',index=False)
