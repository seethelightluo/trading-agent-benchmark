import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2026-07-15'").set_index('date').sort_index() for s in U}
rows=[]
for s,x in D.items():
 prev=x.close.shift(1); intra=-(x.close/x.open-1); clv=-(2*(x.close-x.low)/(x.high-x.low)-1).where((x.high-x.low)!=0); peer=[]
 # peer lead-lag: asset 5d return minus leave-one-out median 5d return
 r=x.close.pct_change(5); allr=pd.concat({q:y.close.pct_change(5) for q,y in D.items()},axis=1); peer=r-(allr.drop(columns=s).median(axis=1))
 rev=-(x.close/x.close.shift(5)-1); mom=x.close/x.close.shift(20)-1
 rows.append(pd.DataFrame({'intra':intra,'clv':clv,'peer':peer,'rev':rev,'mom':mom}))
a=pd.concat(rows).dropna(); print('rows',len(a)); print(a.corr().round(4).to_string())
# date-level IC, regimes, decay
for h in [1,5,10,20]:
 out=[]
 for s,x in D.items():
  f=-(x.close/x.open-1); fr=x.close.shift(-h)/x.close-1
  q=pd.DataFrame({'f':f,'r':fr}).dropna()
  for dt,g in q.groupby(q.index):
   if len(g)>=1: pass
  # collect by calendar date later
  out.append(pd.DataFrame({'date':q.index,'f':q.f.values,'r':q.r.values}))
 z=pd.concat(out); vals=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8:
   c=g.f.corr(g.r,method='spearman')
   if pd.notna(c): vals.append((dt,c))
 zz=pd.DataFrame(vals,columns=['date','ic']).set_index('date')
 print('h',h,'dates',len(zz),'mean',round(zz.ic.mean(),6),'icir',round(zz.ic.mean()/zz.ic.std(ddof=1),6),'hit',round((zz.ic>0).mean(),4))
 if h==1:
  for yr,g in zz.groupby(zz.index.year): print('regime',yr,'n',len(g),'ic',round(g.ic.mean(),6),'icir',round(g.ic.mean()/g.ic.std(ddof=1),5))
# rank turnover
f=pd.concat({s:-(x.close/x.open-1) for s,x in D.items()},axis=1); ranks=f.rank(axis=1,pct=True); print('turn',round(ranks.diff().abs().mean().mean(),6),'coverage',round(f.notna().mean().mean(),6))
