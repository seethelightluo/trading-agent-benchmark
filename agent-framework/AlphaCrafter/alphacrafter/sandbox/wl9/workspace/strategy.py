import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
ENSEMBLE = Path(__file__).parent / "factors" / "factor_ensemble.json"
_calls = 0

def load_ensemble():
    try:
        fs = json.loads(ENSEMBLE.read_text()).get("selected_factors", [])
        if len(fs) <= 10 and abs(sum(float(f["weight"]) for f in fs)-1) < 1e-6: return fs
    except Exception: pass
    return []

def rank(x):
    good=[a for a in ASSETS if np.isfinite(x.get(a,np.nan))]; out={a:.5 for a in ASSETS}
    for i,a in enumerate(sorted(good,key=lambda a:x[a])): out[a]=(i+1)/len(good)
    return out

def hist(a):
    d=get_stock_daily_data(symbol=a,days=450)
    if d is None or len(d)<180: return None
    d=d.sort_values("date").iloc[:-1]
    return np.asarray(d.close,float), np.asarray(d.volume,float) if "volume" in d else np.ones(len(d))

@register_hook
def strategy():
    global _calls
    if _calls%10: _calls+=1; return
    _calls+=1; fs=load_ensemble()
    if not fs: return
    h={a:hist(a) for a in ASSETS}
    if any(x is None for x in h.values()): return
    n=min(len(x[0]) for x in h.values()); c={a:h[a][0][-n:] for a in ASSETS}; v={a:h[a][1][-n:] for a in ASSETS}
    r={a:np.diff(c[a])/np.maximum(c[a][:-1],1e-12) for a in ASSETS}
    r20={a:c[a][-1]/c[a][-21]-1 for a in ASSETS}; r60={a:c[a][-1]/c[a][-61]-1 for a in ASSETS}; r120={a:c[a][-1]/c[a][-121]-1 for a in ASSETS}
    vv={a:max(float(np.std(r[a][-60:],ddof=1)),.004) for a in ASSETS}; market=float(np.mean(list(r60.values()))); breadth=float(np.mean([r20[a]>0 for a in ASSETS])); raw={a:{} for a in ASSETS}
    for a in ASSETS:
        rr=r[a][-60:]; path=float(np.sum(np.abs(rr)))+1e-12; agree=np.mean([np.sign(r20[a]),np.sign(r60[a]),np.sign(r120[a])]); shock=float(np.std(r[a][-10:],ddof=1))/vv[a]
        press=float(np.mean((rr[-20:]-np.mean(rr[-20:]))*(np.maximum(v[a][-20:],1e-12)/max(np.mean(v[a][-60:]),1e-12))))
        raw[a]={"miner_1_20301031_riskadjusted_momentum_60d":r60[a]/vv[a],"miner_2_20301114_compressed_trend_reversal_60d":-r60[a]/vv[a],"miner_2_20301212_efficiency_reversal_60d":-(r60[a]/path)/vv[a],"miner_2_20350315_trend_agreement_reversal_60d":-agree*r60[a]/vv[a],"multimacro_residual_reversal_60d":-(r60[a]-market)/vv[a],"miner_2_20301017_relative_strength_reversal_volscaled_60d":-(r20[a]-float(np.mean(list(r20.values()))))/vv[a],"miner_2_20350426_volume_pressure_fade_20d":-press/vv[a],"miner_2_20350104_volatility_shock_reversal_60d":-shock*r20[a]/vv[a]}
    score={a:0. for a in ASSETS}
    for f in fs:
        q=rank({a:raw[a].get(f["factor_id"],np.nan) for a in ASSETS})
        for a in ASSETS: score[a]+=float(f["weight"])*int(f.get("direction",1))*q[a]
    # Stronger defensive overlay after weak recent validation and bearish consolidation.
    if market<.006 or breadth<.55:
        for a in ("XAU","US10Y","CN10Y"): score[a]+=.20
        for a in ("BTC","ETH","WTI"): score[a]-=.10
    z={a:float(np.clip((score[a]-np.mean(list(score.values())))/max(np.std(list(score.values())),1e-9),-1.5,1.5)) for a in ASSETS}
    w=np.array([max(np.exp(.35*z[a]),.05) for a in ASSETS]); w/=w.sum(); target={a:float(w[i]) for i,a in enumerate(ASSETS)}; forecast={a:float(np.clip(.01*z[a],-.03,.03)) for a in ASSETS}
    rebalance_to_weights(target,forecast_returns=forecast,factor_ids=[f["factor_id"] for f in fs],horizon_days=10)
