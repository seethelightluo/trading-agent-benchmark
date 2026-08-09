import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
rows=[]
for s,d in data.items():
 r=(d.close-d.open)/(d.high-d.low).replace(0,np.nan); fac=r.rolling(5,min_periods=5).mean()
 for h in [1,3,5,10]: rows.append((h,pd.DataFrame({'f':fac,'y':d.close.pct_change(h).shift(-h),'s':s}).dropna()))
def calc(z):
 ics=[]; ns=[]
 for dt,g in z.groupby(z.index):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ics.append(g.f.corr(g.y,method='spearman')); ns.append(len(g))
 a=np.asarray(ics); return len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)
for h in [1,3,5,10]:
 z=pd.concat([x for hh,x in rows if hh==h]); print('h',h,'dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.3f'%(calc(z)))
z=pd.concat([x for hh,x in rows if hh==3]); q=z.pivot_table(index=z.index,columns='s',values='f').rank(axis=1,pct=True); print('turnover',q.diff().abs().mean().mean())
for label,mask in [('pre',z.index<pd.Timestamp('2026-07-16')),('online',z.index>=pd.Timestamp('2026-07-16'))]: print(label,'dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.3f'%calc(z.loc[mask]))
z.assign(date=z.index).to_csv('../persistent/factor_signals_miner_3_20270225_intraday_pressure.csv',index=False)
