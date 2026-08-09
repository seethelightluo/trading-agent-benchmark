import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2026-07-15'").set_index('date').sort_index() for s in U}
rows=[]
for s,x in D.items():
 f=-(x.close/x.open-1); r=x.close.shift(-1)/x.close-1
 rows.append(pd.DataFrame({'date':x.index,'s':s,'f':f,'r':r}))
a=pd.concat(rows,ignore_index=True).dropna(); out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8:
  q=spearmanr(g.f,g.r).statistic
  if pd.notna(q): out.append((dt,q,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); ic=z.ic
turn=[]
for s,x in D.items():
 f=-(x.close/x.open-1)
 turn.append((f.rank(pct=True).diff().abs()>0.05).mean())
# library correlation on common pooled asset-date rows
wide=[]
for s,x in D.items():
 wide.append(pd.DataFrame({'intra':-(x.close/x.open-1),'rev5':-(x.close/x.close.shift(5)-1),'mom20':x.close/x.close.shift(20)-1,'clv':-(2*(x.close-x.low)/(x.high-x.low)-1),'peer':x.close*0}))
w=pd.concat(wide).replace([np.inf,-np.inf],np.nan).dropna(subset=['intra','rev5','mom20','clv'])
c=w.corr()['intra'].drop('intra').abs(); print('dates',len(z),'avg_n',z.n.mean(),'coverage',a.shape[0]/(len(z)*15),'ic',ic.mean(),'icir',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'turn',np.mean(turn)); print('decay')
for h in [5,10,20]:
 rr=[]
 for s,x in D.items(): rr.append(pd.DataFrame({'date':x.index,'f':-(x.close/x.open-1),'r':x.close.shift(-h)/x.close-1}))
 b=pd.concat(rr,ignore_index=True).dropna(); vals=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8: vals.append(g.f.corr(g.r,method='spearman'))
 vals=np.array(vals); print(h,vals.mean(),vals.mean()/vals.std(ddof=1),len(vals))
print('corr',c.to_dict()); print('regimes')
for yr,g in z.groupby(z.index.year): print(yr,len(g),g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1))
