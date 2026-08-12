import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<180:d=get_index_daily_data(s,2800)
 if d is not None and len(d):px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change(); rows=[]; sig=[]
# Soft-threshold dispersion gate: preserve full reversal strength in quiet regimes,
# then linearly attenuate excess dispersion rather than abruptly switching regimes.
for t in range(65,len(P)-11):
 r3=R.iloc[t-2:t+1].sum(); v=R.iloc[t-19:t+1].std(ddof=1); med=r3.median(); disp=float(r3.std(ddof=1))
 hs=[float(R.iloc[k-2:k+1].sum().std(ddof=1)) for k in range(max(65,t-59),t+1)]
 base=float(np.nanmedian(hs)); ratio=disp/base if base>0 else np.nan
 gate=float(np.clip(1-.65*max(ratio-1,0),.35,1))
 f=gate*(-(r3-med)/v.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna();sig.append(f.rename(P.index[t]))
 for h in (1,5,10):
  fw=R.iloc[t+1:t+h+1].sum().reindex(f.index);q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8:rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h];a=z.set_index('date').ic;print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
S=pd.DataFrame(sig);S.to_csv('scripts/miner_1_20300711_soft_dispersion_gate_signal.csv',index_label='date');print('signal_rows',len(S),'instruments',len(U),'available',len(px))
