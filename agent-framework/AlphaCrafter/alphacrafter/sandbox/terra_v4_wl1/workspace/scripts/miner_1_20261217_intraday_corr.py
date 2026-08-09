import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p='scripts/miner_1_20261217_intraday_body3_reversal_signal.csv'; F=pd.read_csv(p,index_col=0,parse_dates=True).reindex(columns=U); cs=[]
for q in Path('scripts').glob('*signal.csv'):
 try:
  x=pd.read_csv(q,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U); c=F.stack().corr(x.stack())
  if pd.notna(c): cs.append((abs(c),q.name,c))
 except Exception: pass
print(sorted(cs,reverse=True)[:10])
