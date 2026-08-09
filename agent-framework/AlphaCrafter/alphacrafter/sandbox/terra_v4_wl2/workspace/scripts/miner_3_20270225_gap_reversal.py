import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-02-24')
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); d=d.loc[:END]
 gap=d.open/d.close.shift(1)-1
 for h in [1,3,5,10]: rows.append((h,pd.DataFrame({'f':-gap,'y':d.close.pct_change(h).shift(-h),'s':s}).dropna()))
def calc(z):
 a=[]; ns=[]
 for dt,g in z.groupby(z.index):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:a.append(g.f.corr(g.y,method='spearman'));ns.append(len(g))
 a=np.asarray(a);return len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)
for h in [1,3,5,10]:
 z=pd.concat([x for hh,x in rows if hh==h]); print('h',h,calc(z))
 for lab,m in [('pre',z.index<pd.Timestamp('2026-07-16')),('online',z.index>=pd.Timestamp('2026-07-16'))]:print(lab,calc(z.loc[m]))
z=pd.concat([x for hh,x in rows if hh==1]);q=z.pivot_table(index=z.index,columns='s',values='f').rank(axis=1,pct=True);print('turn',q.diff().abs().mean().mean(),'coverage',len(z)/((z.index.max()-z.index.min()).days*15/7*5))
z.assign(date=z.index).to_csv('../persistent/factor_signals_miner_3_20270225_gap_reversal.csv',index=False)
