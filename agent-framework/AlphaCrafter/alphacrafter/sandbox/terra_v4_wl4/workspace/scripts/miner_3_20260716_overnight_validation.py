import pandas as pd, numpy as np, glob, json
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2026-07-15'").set_index('date').sort_index() for s in U}
def eval_factor(kind,h):
 rows=[]
 for s,x in D.items():
  prev=x.close.shift(1)
  f=-(x.open/prev-1) if kind=='gap' else -(x.close/x.open-1)
  r=x.close.shift(-h)/x.close-1
  rows.append(pd.DataFrame({'date':x.index,'s':s,'f':f.values,'r':r.values}))
 a=pd.concat(rows).dropna(subset=['f','r']); out=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8:
   ic=g.f.corr(g.r,method='spearman')
   if pd.notna(ic): out.append((dt,ic,len(g)))
 z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
 return z
for k in ['gap','intra']:
 print('\n',k)
 for h in [1,5,10,20]:
  z=eval_factor(k,h); print(h,'dates',len(z),'meanN',round(z.n.mean(),2),'IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
 z=eval_factor(k,1)
 # factor coverage and turnover on asset-date observations
 cov=[]; turns=[]
 for s,x in D.items():
  prev=x.close.shift(1); f=-(x.open/prev-1) if k=='gap' else -(x.close/x.open-1)
  cov.append(f.notna().mean()); turns.append((f.rank(pct=True).diff().abs()>0.05).mean())
 print('coverage',round(np.mean(cov),4),'turnover_proxy',round(np.mean(turns),4))
 for yr,g in z.groupby(z.index.year): print('regime',yr,round(g.ic.mean(),5),len(g))
# pooled rank correlations to existing simple factor definitions on aligned rows
allrows=[]
for s,x in D.items():
 prev=x.close.shift(1); gap=-(x.open/prev-1); intra=-(x.close/x.open-1); rev5=-(x.close/x.close.shift(5)-1); mom20=x.close/x.close.shift(20)-1
 allrows.append(pd.DataFrame({'gap':gap,'intra':intra,'rev5':rev5,'mom20':mom20}))
a=pd.concat(allrows).dropna(); print('corr',a.corr().round(4).to_dict()['gap'])
