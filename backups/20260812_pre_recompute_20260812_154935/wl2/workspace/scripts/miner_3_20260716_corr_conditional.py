import numpy as np,pandas as pd,json,glob
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r3=p.pct_change(3); cand=(-r3).where(r3<0)
for path in glob.glob('factors/*.json'):
 try:
  j=json.load(open(path)); e=j.get('calculation',{}).get('expression','')
  if 'pct_change' in e and '5' in e: f=-p.pct_change(5)
  elif 'pct_change' in e and '3' in e: f=-p.pct_change(3)
  else: continue
  x=pd.concat([cand.stack().rename('c'),f.stack().rename('f')],axis=1).dropna();print(path,round(x.c.corr(x.f),5))
 except Exception as e: print(e)
