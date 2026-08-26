import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date') for s in U}
def calc(s,h):
 d=D[s]; c=d['close'].astype(float); r=c.pct_change(); down=r.where(r<0,0).rolling(40).std(); up=r.where(r>0,0).rolling(40).std()
 raw=(c/c.shift(40)-1)/(down+1e-8)*(1+up/(down+1e-8)).clip(.5,2.5)
 return pd.DataFrame({'date':d.date,'f':-raw,'fr':c.shift(-h)/c-1})
for h in [5,10,20,40]:
 z=pd.concat([calc(s,h).assign(s=s) for s in U]).dropna(); vals=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8: vals.append(g['f'].corr(g['fr'],method='spearman'))
 q=pd.Series(vals).dropna();print(f'h={h} dates={len(q)} mean_n={z.groupby("date").size().mean():.2f} IC={q.mean():.6f} ICIR={q.mean()/q.std():.6f} hit={(q>0).mean():.4f}')
z=pd.concat([calc(s,10).assign(s=s) for s in U]).pivot(index='date',columns='s',values='f')
z.to_csv('scripts/miner_2_20291213_tail_resilience_signal.csv')
print('coverage',z.notna().mean().mean(),'turnover',z.rank(axis=1,pct=True).diff().abs().mean().mean())
