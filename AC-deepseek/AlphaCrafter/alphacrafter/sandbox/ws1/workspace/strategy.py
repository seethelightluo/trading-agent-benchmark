"""Defensive cross-asset ensemble, Screener 2035-12-20.
Completed daily bars only; one atomic fully-invested rebalance per decision."""
from math import isfinite
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
    get_index_daily_data, rebalance_to_weights, register_hook)

# Exact admitted M=10 weights: defensive/tail 56%, recovery/persistence/trend 29%, macro 15%.
FW = (.17, .15, .13, .11, .10, .08, .08, .07, .06, .05)
DEF = {"XAU", "US10Y", "CN10Y"}
DEF_W, CAP, TOP = .12, .15, 15

def stock(a, n=145):
    try: return get_stock_daily_data(a, days=n)
    except Exception: return None

def index(a, n=100):
    try: return get_index_daily_data(a, days=n)
    except Exception: return None

def ranks(values, assets):
    valid = sorted((float(v), a) for a, v in values.items() if v is not None and isfinite(float(v)))
    out = {a: .5 for a in assets}
    for i, (_, a) in enumerate(valid): out[a] = i / max(1, len(valid)-1)
    return out

def loading_contraction(y, x):
    z = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().tail(60)
    def beta(n):
        q=z.tail(n); var=float(q.x.var())
        return float(q.y.cov(q.x)/var) if len(q)>=12 and var>1e-14 else None
    new, old = beta(20), beta(60)
    return old-new if new is not None and old is not None else None

def capped_normalize(w, pref):
    for _ in range(40):
        excess=sum(max(0., x-CAP) for x in w.values())
        w={a:min(CAP, max(0., x)) for a, x in w.items()}
        room=[a for a,x in w.items() if x<CAP-1e-12]
        if excess < 1e-12 or not room: break
        den=sum(max(0.,pref.get(a,0.)) for a in room)
        for a in room: w[a] += excess*(max(0.,pref.get(a,0.))/den if den else 1/len(room))
    total=sum(w.values())
    return {a:x/total for a,x in w.items()}

@register_hook
def strategy_hook():
    assets=list(get_account_dict()["watch_list"])
    frames={a:stock(a) for a in assets}
    closes={a:(f.close.astype(float) if f is not None and "close" in f and len(f)>=62 else None) for a,f in frames.items()}
    usable=[c.pct_change().rename(a) for a,c in closes.items() if c is not None]
    panel=pd.concat(usable,axis=1,join="inner").dropna().tail(100) if len(usable)>=8 else pd.DataFrame()
    if len(panel)<61:
        rebalance_to_weights(
            {a:1/len(assets) for a in assets},
            forecast_returns={a:0.0 for a in assets},
            horizon_days=10,
        ); return
    market=panel.mean(axis=1); mv=float(market.var())
    residual={a:panel[a]-(float(panel[a].cov(market)/mv)*market if mv>1e-14 else 0.) for a in panel}
    vf=index("VIX"); vix=vf.close.astype(float).pct_change() if vf is not None and "close" in vf else None
    oil,copper=closes.get("WTI"),closes.get("COPPER")
    infl=(oil.pct_change()+copper.pct_change())*.5 if oil is not None and copper is not None else None
    stress=vix.reindex(infl.index).clip(lower=0)*infl if vix is not None and infl is not None else None
    commodity_rel=oil.pct_change()-copper.pct_change() if oil is not None and copper is not None else None
    wealth=(1+market).cumprod(); mdd=wealth/wealth.rolling(60).max()-1
    basket=panel[[a for a in DEF if a in panel]].mean(axis=1); dispersion=panel.std(axis=1)
    high_disp=float(dispersion.tail(20).mean()) >= float(dispersion.tail(60).median())
    beta={}; resilience={}; recovery={}; lpm={}; draw={}; macro={}; acorr={}; conditional={}; commodity={}; trend={}; vol={}
    for a,e in residual.items():
        ret=panel[a]; vol[a]=float(ret.tail(20).std())
        z=pd.concat([e.rename("e"),basket.rename("b"),market.rename("m")],axis=1).dropna()
        def downbeta(n):
            q=z.tail(n); q=q[q.m<0]; var=float(q.b.var())
            return float(q.e.cov(q.b)/var) if len(q)>=5 and var>1e-14 else None
        new,old=downbeta(20),downbeta(60); beta[a]=old-new if new is not None and old is not None else None
        recovery[a]=float(ret.tail(20).clip(upper=0).mean())-float(ret.iloc[:-20].tail(40).clip(upper=0).mean())
        resilience[a]=loading_contraction(e,vix.clip(lower=0)**2) if vix is not None else None
        macro[a]=loading_contraction(e,stress) if stress is not None else None
        oc=float(ret.iloc[:-20].tail(40).corr(mdd.iloc[:-20].tail(40))); nc=float(ret.tail(20).corr(mdd.tail(20)))
        draw[a]=oc-nc if isfinite(oc) and isfinite(nc) else None
        neg=ret.tail(60).clip(upper=0); lpm[a]=-float((neg*neg).mean()**.5)
        conditional[a]=recovery[a] if (high_disp and float(market.tail(20).mean())<0) else None
        na,oa=float(e.tail(20).autocorr(1)),float(e.iloc[:-20].tail(40).autocorr(1))
        acorr[a]=na-oa if isfinite(na) and isfinite(oa) else None
        commodity[a]=loading_contraction(e,commodity_rel) if commodity_rel is not None else None
        trend[a]=float(ret.tail(20).mean())/max(vol[a],.003)
    factors=(beta,resilience,draw,lpm,recovery,acorr,macro,commodity,trend,conditional)
    score={a:sum(x*ranks(f,assets)[a] for x,f in zip(FW,factors)) for a in assets}
    selected=sorted(assets,key=lambda a:(score[a],a),reverse=True)[:TOP]
    pref={a:.4*(2-i/max(1,TOP-1))/max(vol.get(a,.03) or .03,.003)+.6 for i,a in enumerate(selected) if a not in DEF}
    nondef=[a for a in assets if a not in DEF]; den=sum(pref.get(a,0.) for a in nondef)
    raw={a:(DEF_W if a in DEF else .64*pref.get(a,0.)/den) for a in assets} if den else {a:1/len(assets) for a in assets}
    weights=capped_normalize(raw,pref)
    weights[assets[-1]] += 1-sum(weights.values())
    score_values = [float(score[a]) for a in assets]
    score_mean = sum(score_values) / len(score_values)
    score_std = (sum((value - score_mean) ** 2 for value in score_values) / len(score_values)) ** .5
    return_scale = float(panel.tail(252).std(axis=1, ddof=0).median()) if len(panel) else .01
    if not isfinite(return_scale) or return_scale <= 0: return_scale = .01
    forecast_returns = {
        a: ((float(score[a]) - score_mean) / max(score_std, 1e-12)) * return_scale
        for a in assets
    }
    try:
        ensemble = json.loads((Path(__file__).parent / "factor_ensemble.json").read_text())
        factor_ids = [str(item["factor_id"]) for item in ensemble.get("selected_factors", []) if isinstance(item, dict) and item.get("factor_id")]
    except (OSError, ValueError, TypeError):
        factor_ids = []
    rebalance_to_weights(
        weights,
        forecast_returns=forecast_returns,
        factor_ids=factor_ids[:10],
        horizon_days=10,
    )
