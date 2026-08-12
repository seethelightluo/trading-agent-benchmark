import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, get_index_daily_data, get_account_dict, rebalance_to_weights

U = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
F = [
    "miner_1_20281116_defensive_relative_lead_20d", "macro_stress_resilience_20d",
    "stable_asymmetry_40_60", "breadth_vol_quality_40d",
    "miner_2_20300221_accel_persistence_trend", "directional_payoff_efficiency_30d",
    "miner_1_20291227_short_reversal_volscaled_20d",
    "miner_3_20280907_downside_asymmetry_quality_30d",
    "miner_1_20281130_residual_volscaled_trend_20d"
]
W = np.array([.19,.16,.15,.13,.13,.09,.06,.05,.04])
_gate = 0

def rank(values, names):
    order = sorted(names, key=lambda s: values.get(s, 0.0))
    return {s: (order.index(s)+1.0)/len(order) for s in U}

def capped(raw, cap=.16):
    # Waterfall cap, then normalize; all 15 tradable assets remain represented.
    r = {s:max(float(raw.get(s, 0.0)), .001) for s in U}; out={s:0.0 for s in U}
    active=set(U); left=1.0
    while active:
        z=sum(r[s] for s in active)
        hit=[s for s in active if left*r[s]/z > cap]
        if not hit:
            for s in active: out[s]=left*r[s]/z
            break
        for s in hit: out[s]=cap; left-=cap; active.remove(s)
    z=sum(out.values())
    return {s:out[s]/z for s in U}

@register_hook
def cross_asset_strategy():
    global _gate
    if _gate:
        _gate -= 1
        return
    prices={}; returns={}; vols={}
    for s in U:
        d=get_stock_daily_data(symbol=s, days=280)
        if d is None or len(d)<125: continue
        c=np.asarray(d.sort_values('date')['close'],dtype=float)[:-1]
        if len(c)>=120 and np.all(np.isfinite(c)) and np.all(c>0):
            prices[s]=c; returns[s]=c[1:]/c[:-1]-1; vols[s]=max(float(np.std(returns[s][-40:])),.008)
    names=[s for s in U if s in prices]
    if len(names)<12: return
    hs=(10,20,30,60)
    R={s:{h:prices[s][-1]/prices[s][-(h+1)]-1 for h in hs} for s in names}
    peer=np.mean([returns[s][-20:] for s in names],axis=0)
    raw={s:[0.0]*9 for s in names}
    for s in names:
        med20=np.median([R[t][20] for t in names if t!=s]); med60=np.median([R[t][60] for t in names if t!=s])
        beta=np.cov(returns[s][-20:],peer)[0,1]/max(np.var(peer),1e-8)
        neg=returns[s][-30:][returns[s][-30:]<0]; pos=returns[s][-30:][returns[s][-30:]>0]
        down=float(np.mean(np.abs(neg))) if len(neg) else .01
        payoff=(float(np.mean(pos)) if len(pos) else 0.0)/max(down,.01)
        persistence=np.mean(returns[s][-30:]>=0)-.5*np.mean(returns[s][-30:]<-.015)
        trend=R[s][20]/max(vols[s],.01)
        resid=(R[s][20]-beta*np.mean([R[t][20] for t in names]))/max(vols[s],.01)
        raw[s]=[R[s][20]-med20+.35*(R[s][60]-med60),
                 .5*R[s][20]/max(vols[s],.01)+.5*R[s][60]/max(vols[s],.01),
                 payoff-.5*vols[s], persistence-vols[s],
                 persistence+.35*trend, payoff+.25*trend, -returns[s][-2]/max(vols[s],.01),
                 payoff-.75*np.mean(np.minimum(returns[s][-30:],0)**2), resid]
    ranks=[rank({s:raw[s][j] for s in names},names) for j in range(9)]
    score={s:sum(W[j]*(ranks[j][s]-.5) for j in range(9)) for s in U}
    # Defensive tilt is applied only in the stated high-risk regime.
    breadth=np.mean([R[s][20]>0 for s in names]); market=R.get('SPX',{}).get(20,float(np.median([R[s][20] for s in names])))
    vd=get_index_daily_data(symbol='VIX',days=55); vix=0.0
    if vd is not None and len(vd)>22:
        vc=np.asarray(vd.sort_values('date')['close'],dtype=float)[:-1]
        if len(vc)>=22 and np.all(np.isfinite(vc)): vix=float(vc[-1])
    defensive=market<-.05 or breadth<.40 or np.mean(list(vols.values()))>.022 or vix>=20
    tilt={s:1.0 for s in U}
    if defensive: tilt.update({'XAU':1.20,'US10Y':1.12,'CN10Y':1.08,'BTC':.90,'ETH':.88,'WTI':.92,'COPPER':.94})
    target=capped({s:tilt[s]*(.51+score[s])/max(vols.get(s,.02),.008)**.15 for s in U})
    account=get_account_dict(); assets=max(float(account.get('total_assets',0)),1.0)
    old={s:0.0 for s in U}
    for p in account.get('positions',[]):
        s=p.get('symbol')
        if s in old and float(p.get('quantity',0))>0: old[s]=max(float(p.get('market_value',0)),0)/assets
    if sum(old.values())>.001:
        target={s:.25*target[s]+.75*old[s] for s in U}; z=sum(target.values()); target={s:target[s]/z for s in U}
    forecast={s:float(.012*score[s]) for s in U}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=F, horizon_days=10)
    _gate=9

cross_asset_strategy
