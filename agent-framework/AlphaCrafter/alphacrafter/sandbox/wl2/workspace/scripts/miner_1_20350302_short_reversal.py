import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def main():
 wl=get_account_dict().get('watch_list',[]); syms=[s for s in U if not wl or s in wl]; px={}
 for s in syms:
  d=get_stock_daily_data(s,days=5000)
  if d is not None and len(d): px[s]=d.set_index('date').close.astype(float)
 p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std()*np.sqrt(20)
 # Short-term reversal, risk normalized and damped when the cross-sectional dispersion is low.
 rev=-p.pct_change(5).div(vol.replace(0,np.nan)); disp=r.rolling(20).std().mean(axis=1); scale=(disp/disp.rolling(120).median()).clip(.5,2)
 f=rev.mul(scale,axis=0); rec=[]
 for i in range(100,len(p)-40):
  x=f.iloc[i].dropna()
  if len(x)<8: continue
  row={'date':p.index[i],'n':len(x)}
  for h in [5,10,20,40]: row['ic'+str(h)]=x.corr((p.iloc[i+h]/p.iloc[i]-1).reindex(x.index),method='spearman')
  rec.append(row)
 q=pd.DataFrame(rec); print('dates',len(q),'mean_n',q.n.mean(),'period',q.date.min(),q.date.max())
 for h in [5,10,20,40]:
  z=q['ic'+str(h)].dropna(); print(h,'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 print('coverage %.4f turnover_proxy %.4f'%(f.notna().sum(axis=1).mean()/len(syms),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
 q.to_csv('../persistent/miner_1_20350302_short_reversal_ic.csv',index=False); f.to_csv('../persistent/miner_1_20350302_short_reversal_signal.csv')
if __name__=='__main__': main()
