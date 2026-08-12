import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<200:d=get_index_daily_data(s,1800)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change();w=60
down=R.clip(upper=0).pow(2).rolling(w).mean();up=R.clip(lower=0).pow(2).rolling(w).mean();sig=(-(down/(up+1e-8))).shift(1);fwd=P.pct_change().shift(-1)
rows=[]
for dt in sig.index:
 for s in P.columns:
  if np.isfinite(sig.loc[dt,s]) and np.isfinite(fwd.loc[dt,s]):rows.append({'date':dt,'symbol':s,'signal':sig.loc[dt,s],'forward_return':fwd.loc[dt,s]})
pd.DataFrame(rows).to_csv('scripts/miner_2_20291004_downside_asymmetry_60d_signal.csv',index=False);print('saved',len(rows),'rows',pd.DataFrame(rows).date.nunique(),'dates')
