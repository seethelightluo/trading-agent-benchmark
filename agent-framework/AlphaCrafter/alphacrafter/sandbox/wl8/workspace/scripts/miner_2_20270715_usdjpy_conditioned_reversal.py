import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-14')
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 x['r3']=x.close/x.close.shift(3)-1; x['f1']=x.close.shift(-1)/x.close-1; x['f5']=x.close.shift(-5)/x.close-1; x['symbol']=s; rows.append(x[['date','symbol','r3','f1','f5']])
z=pd.concat(rows)
m=pd.read_csv('../persistent/index_data/USDJPY.csv'); m.date=pd.to_datetime(m.date); m=m[m.date<=END].sort_values('date').set_index('date')
# lagged standardized 5-day FX move; magnitude captures cross-asset risk shock without directional assumptions
move=m.close.pct_change(5); vol=move.rolling(60,min_periods=30).std(); state=(move/vol).shift(1).clip(-2,2)
z=z.merge(state.rename('fx_state'),left_on='date',right_index=True,how='left')
z['sig']=-z.r3*(1+0.5*z.fx_state.abs())
def calc(df,h):
 vals=[]; ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1:
   vals.append(spearmanr(g.sig,g[h]).statistic); ns.append(len(g))
 v=np.asarray(vals); return len(v),len(df),float(np.mean(ns)),float(v.mean()),float(v.mean()/v.std(ddof=1)),float((v>0).mean())
for h in ['f1','f5']:
 n,r,an,ic,ir,hit=calc(z,h); print(h,'dates',n,'rows',r,'avg_n',round(an,2),'ic',round(ic,6),'icir',round(ir,6),'hit',round(hit,4))
print('coverage',round(z.sig.notna().mean(),4))
for q,c in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027)]:
 n,r,an,ic,ir,hit=calc(z[c],'f1'); print(q,'dates',n,'ic',round(ic,6),'icir',round(ir,6),'hit',round(hit,4))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_2_20270715_usdjpy_conditioned_reversal_signal.csv',index=False)
