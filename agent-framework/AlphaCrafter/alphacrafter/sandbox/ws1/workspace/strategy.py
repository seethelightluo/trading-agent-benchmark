import json
import math
import numpy as np
from alphacrafter.sim.utils import register_hook, add_order, get_stock_daily_data, get_account_dict
UNIVERSE=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DEFENSIVE={"XAU","US10Y","CN10Y"}; CADENCE=10; MIN_W=.015; MAX_W=.16
FACTORS={"miner_3_clv_1d":.3395,"peer_median_leadlag_5d":.2612,"short_term_reversal_5d":.2012,"miner_2_risk_adjusted_momentum_20d":.1981}
try:
 with open("factors/factor_ensemble.json",encoding="utf-8") as f:
  q=json.load(f).get("selected_factors",[])[:10]
 if q: FACTORS={x["factor_id"]:float(x.get("weight",0))*float(x.get("direction",1)) for x in q}
except Exception: pass
_day=0
def rank(a):
 g=sorted(v for v in a.values() if np.isfinite(v)); n=len(g)
 return {s:(sum(v<=a[s] for v in g)/n if n and np.isfinite(a[s]) else .5) for s in UNIVERSE}
def alloc(sc):
 w={s:MIN_W for s in UNIVERSE}; free=set(UNIVERSE); rem=1-len(UNIVERSE)*MIN_W
 while free and rem>1e-12:
  z=sum(max(sc[s],1e-8) for s in free); cap=[s for s in free if rem*max(sc[s],1e-8)/z>=MAX_W-MIN_W]
  if not cap:
   for s in free:w[s]+=rem*max(sc[s],1e-8)/z
   break
  for s in cap:w[s]=MAX_W; rem-=MAX_W-MIN_W; free.remove(s)
 z=sum(w.values()); return {s:w[s]/z for s in UNIVERSE}
@register_hook
def cross_asset_allocator():
 global _day
 _day+=1
 if (_day-1)%CADENCE:return
 ac=get_account_dict(); pos={p.get("symbol"):p for p in ac.get("positions",[])}; close={}; vol={}; clv={}; r5={}
 for s in UNIVERSE:
  d=get_stock_daily_data(symbol=s,days=125)
  if d is None or len(d)<45:continue
  d=d.sort_values("date"); p=np.asarray(d["close"],float)
  if len(p)<42 or np.any(~np.isfinite(p)) or np.any(p<=0):continue
  close[s]=p; vol[s]=max(float(np.std(p[1:]/p[:-1]-1)[-20:])*math.sqrt(252),.05) if False else max(float(np.std((p[1:]/p[:-1]-1)[-20:])*math.sqrt(252)),.05)
  hi,lo=float(d.iloc[-1]["high"]),float(d.iloc[-1]["low"]); clv[s]=(2*p[-1]-hi-lo)/(hi-lo) if hi>lo else 0; r5[s]=p[-1]/p[-6]-1
 med=float(np.nanmedian(list(r5.values()))) if r5 else 0
 raw={"miner_3_clv_1d":clv,"peer_median_leadlag_5d":{s:r5.get(s,np.nan)-med for s in UNIVERSE},"short_term_reversal_5d":{s:-r5.get(s,np.nan) for s in UNIVERSE},"miner_2_risk_adjusted_momentum_20d":{s:(.5*(close[s][-1]/close[s][-21]-1)+.5*(close[s][-1]/close[s][-41]-1))/vol[s] if s in close else np.nan for s in UNIVERSE}}
 rk={k:rank(v) for k,v in raw.items()}; iv=rank({s:1/vol.get(s,.5) for s in UNIVERSE}); sp=close.get("SPX"); bear=sp is not None and sp[-1]/sp[-31]<.995
 sc={s:max(.78*sum(FACTORS.get(k,0)*rk[k][s] for k in raw)+.22*iv[s]*(.88 if vol.get(s,.5)>.75 else 1)+(.12 if bear and s in DEFENSIVE else 0),1e-8) for s in UNIVERSE}; wt=alloc(sc); total=float(ac.get("total_assets",0) or 0)
 price={s:(float(close[s][-1]) if s in close else float((pos.get(s) or {}).get("current_price",0) or 0)) for s in UNIVERSE}
 for s in UNIVERSE:
  if price[s]<=0:continue
  cur=float((pos.get(s) or {}).get("quantity",0) or 0); av=float((pos.get(s) or {}).get("available_quantity",cur) or 0); q=int(min(max(cur-total*wt[s]/price[s],0),max(av,0)))
  if q>0:add_order(symbol=s,order_type="SELL",price=price[s],quantity=q)
 for s in UNIVERSE:
  if price[s]<=0:continue
  cur=float((pos.get(s) or {}).get("quantity",0) or 0); q=int(max(total*wt[s]/price[s]-cur,0))
  if q>0:add_order(symbol=s,order_type="BUY",price=price[s],quantity=q)
