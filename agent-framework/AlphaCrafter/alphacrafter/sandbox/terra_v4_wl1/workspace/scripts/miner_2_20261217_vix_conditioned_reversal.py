import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-16')
base='../persistent/stock_data'; macro='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].loc[:cut]
v5=v.pct_change(5).reindex(P.index).ffill(); r5=P.pct_change(5); sh=v5.clip(-.5,.5)
# In elevated VIX-trend conditions use full reversal; otherwise damp it to reduce noise.
f=(-r5.mul(1+sh,axis=0)).where(sh>0,-r5*.5)
rows=[]
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; out=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: out.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1: print('years',[(int(y),round(g.mean(),6),len(g)) for y,g in ic.groupby(ic.index.year)])
# Persist a recoverable, date-aligned signal artifact for audit.
f.to_csv('scripts/miner_2_20261217_vix_conditioned_reversal_signal.csv',index_label='date')
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5),'high_vix_dates',int((sh>0).sum()))
print('assets',len(U),'price_rows',len(P),'cutoff',cut.date(),'signal_artifact','scripts/miner_2_20261217_vix_conditioned_reversal_signal.csv')
