import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr

END=pd.Timestamp('2030-01-09')
files=glob.glob('../persistent/stock_data/*.csv')
frames={}
for f in files:
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date')
 d=d[d.date<=END].set_index('date')
 frames[s]=d
# Path-efficiency trend: lagged 20d return divided by total absolute path, with a mild 60d confirmation
rows=[]
for s,d in frames.items():
 r=d.close.pct_change()
 ret20=d.close.pct_change(20)
 path20=r.abs().rolling(20).sum()
 confirm=d.close.pct_change(60)
 # signal known at t-1; construct at t then shift; rank correlation uses signal date t and future t+10
 sig=(ret20/path20)*(1+0.5*np.tanh(confirm/0.20))
 fwd=d.close.shift(-10)/d.close-1
 z=pd.DataFrame({'signal':sig.shift(1),'fwd':fwd},index=d.index)
 z['symbol']=s; rows.append(z.reset_index())
x=pd.concat(rows).rename(columns={'index':'date'})
# date-wise IC, at least 8 names
out=[]
for dt,g in x.groupby('date'):
 g=g.replace([np.inf,-np.inf],np.nan).dropna()
 if len(g)>=8:
  ic=spearmanr(g.signal,g.fwd).statistic
  out.append((dt,ic,len(g)))
ic=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
mean=ic.ic.mean(); sd=ic.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
# paper ICIR convention from prior likely mean/std (not annualized)?? report both
print('dates',len(ic),'range',ic.index.min().date(),ic.index.max().date(),'avg_n',ic.n.mean())
print('IC',mean,'ICIR_daily',mean/sd,'ICIR_annualized',icir,'hit', (ic.ic>0).mean())
for a,b in [('2020','2023'),('2024','2026-07-15'),('2026-07-16','2028-12-31'),('2029-01-01','2030-01-09')]:
 q=ic.loc[a:b]
 if len(q): print('regime',a,b,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
# turnover: rank signal changes, average normalized rank distance on common dates
wide=x.pivot(index='date',columns='symbol',values='signal').replace([np.inf,-np.inf],np.nan)
ranks=wide.rank(axis=1,pct=True); changes=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna()
print('coverage',x.signal.notna().mean(),'turnover_proxy',changes.mean())
# save artifact
art=x[['date','symbol','signal']].dropna().sort_values(['date','symbol'])
art.to_csv('scripts/miner_2_20300110_path_efficiency_trend_signal.csv',index=False)
print('artifact',len(art),'rows')
