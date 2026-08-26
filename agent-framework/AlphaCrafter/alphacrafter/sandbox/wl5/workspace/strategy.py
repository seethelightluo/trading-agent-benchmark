import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

U = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
_skip = 0

def load_factors():
    try:
        p = Path(__file__).parent / "factor_ensemble.json"
        fs = json.loads(p.read_text()).get("selected_factors", [])
        if 0 < len(fs) <= 10 and abs(sum(float(f["weight"]) for f in fs)-1) < 1e-6:
            return fs
    except Exception:
        pass
    return []

def rank(x):
    a = sorted(U, key=lambda s:(float(x[s]),s))
    return {s:(i-7.0)/14.0 for i,s in enumerate(a)}

def bounded(raw):
    lo, hi = .025, .20
    w={s:lo for s in U}; free=set(U); rem=1-lo*len(U)
    while free:
        z=sum(max(float(raw[s]),1e-10) for s in free)
        add={s:rem*max(float(raw[s]),1e-10)/z for s in free}
        cap={s for s in free if add[s]>hi-lo}
        if not cap:
            for s in free:w[s]+=add[s]
            break
        for s in cap:w[s]=hi
        rem-=len(cap)*(hi-lo); free-=cap
    z=sum(w.values()); return {s:w[s]/z for s in U}

@register_hook
def cross_asset_strategy():
    global _skip
    if _skip:
        _skip-=1; return
    factors=load_factors()
    if not factors:return
    data={}
    for s in U:
        d=get_stock_daily_data(symbol=s,days=300)
        if d is None or len(d)<140:return
        data[s]=d.sort_values("date")
    c={s:np.maximum(np.asarray(data[s]["close"],float),1e-12) for s in U}
    lr={s:np.diff(np.log(c[s])) for s in U}
    v20={s:max(float(np.std(lr[s][-20:])),.008) for s in U}
    ret=lambda s,n:c[s][-1]/c[s][-1-n]-1
    r20={s:ret(s,20) for s in U}; r40={s:ret(s,40) for s in U}; r60={s:ret(s,60) for s in U}; r90={s:ret(s,90) for s in U}
    market=np.mean(np.array([lr[s][-45:] for s in U]),axis=0)
    residual={}
    for s in U:
        a=lr[s][-45:]; beta=float(np.cov(a[-40:],market[-40:])[0,1]/max(np.var(market[-40:]),1e-12))
        residual[s]=-float(np.sum((a-beta*market)[-10:]))
    score={s:0. for s in U}
    for f in factors:
        fid,wt=f["factor_id"],float(f["weight"]); direction=int(f.get("direction",1))
        if fid=="miner_1_20331027_inverse_trend60_vol20": x={s:-r60[s]/v20[s] for s in U}
        elif fid=="miner_2_20330915_long_persistence_reversal_90d": x={s:-r90[s]/max(float(np.std(lr[s][-45:])),.008) for s in U}
        elif fid=="miner_2_20330901_trend_persistence_reversal_60d": x={s:-r60[s]/max(float(np.std(lr[s][-45:])),.008) for s in U}
        elif fid=="miner_1_20290823_risk_adjusted_intermediate_momentum_20d": x={s:r20[s]/v20[s] for s in U}
        elif fid=="miner_2_20330818_downside_quality_reversal_40d":
            x={}
            for s in U:
                rr=lr[s][-60:]; down=float(np.std(np.minimum(rr,0))); total=float(np.std(rr))
                x[s]=-(r40[s]/max(down,.008))*(1+.25*(1-down/max(total,.008)))
        elif fid=="miner_2_20331208_vol_adjusted_momentum40_inverse": x={s:-r40[s]/max(float(np.std(lr[s][-40:])),.008) for s in U}
        elif fid=="miner_3_20330526_beta_instability_residual_reversal": x={s:residual[s]/max(float(np.std(lr[s][-20:])),.008) for s in U}
        else:return
        q=rank(x)
        for s in U:score[s]+=wt*direction*q[s]
    breadth=np.mean([r20[s]>0 for s in U]); dispersion=float(np.std(list(r20.values())))
    risk=min(1.,dispersion/.08)
    raw={s:max(score[s]+.55,1e-8)/v20[s]**.60 for s in U}
    if breadth<.5 or risk>.55:
        for s in DEFENSIVE:raw[s]*=1.5+.5*risk
    target=bounded(raw)
    sd=max(float(np.std(list(score.values()))),1e-9); avg=float(np.mean(list(score.values())))
    forecast={s:float(.025*(score[s]-avg)/sd) for s in U}
    for s in DEFENSIVE:forecast[s]+=float(.004*(1+risk))
    rebalance_to_weights(target,forecast_returns=forecast,factor_ids=[f["factor_id"] for f in factors],horizon_days=10)
    _skip=9
# Full 15-asset, long-only, fractional-compatible target; no cash sleeve.
