import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-10-20'); rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 x['r5']=x.close/x.close.shift(5)-1; x['f1']=x.close.shift(-1)/x.close-1; x['f5']=x.close.shift(-5)/x.close-1; x['symbol']=s; rows.append(x[['date','symbol','r5','f1','f5']])
z=pd.concat(rows)
bench=z[z.symbol=='SPX'][['date','r5']].rename(columns={'r5':'br5'}); z=z.merge(bench,on='date',how='left')
z['sig']=-(z.r5-z.br5)
def calc(df,h):
 vals=[]; ns=[]
 for d,g in df.dropna(subset=['sig',h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1: vals.append(spearmanr(g.sig,g[h]).statistic); ns.append(len(g))
 a=np.array(vals); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
for h in ['f1','f5']:
 n,an,ic,ir,hit=calc(z,h); print(h,'dates',n,'avg_n',round(an,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4))
print('rows',len(z),'coverage',round(z.sig.notna().mean(),4))
for q,c in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027),('last180',z.date>=END-pd.Timedelta(days=180))]:
 n,an,ic,ir,hit=calc(z[c],'f1'); print(q,'dates',n,'avg_n',round(an,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4))
z[['date','symbol','sig']].dropna().to_csv('scripts/miner_2_20271021_residual_reversal_signal.csv',index=False)
