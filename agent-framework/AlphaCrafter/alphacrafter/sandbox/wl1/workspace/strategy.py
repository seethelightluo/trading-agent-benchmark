import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, get_account_dict, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
FACTOR_IDS = [
    "miner_3_20330610_medium_relative_contrarian_60_15d",
    "miner_1_20330204_risk_adjusted_residual_contrarian_40d",
    "miner_1_20320624_path_stability_lead_10d",
    "miner_1_20281116_defensive_relative_lead_20d",
    "miner_3_20310904_recovery_pullback_20d",
    "macro_stress_resilience_20d",
    "miner_3_20280907_downside_asymmetry_quality_30d",
    "miner_1_20281005_vix_shock_resilient_momentum_20d",
]
FACTOR_WEIGHTS = np.array([.20, .16, .15, .15, .12, .08, .08, .06])
WAIT = 0

def rank01(v):
    a = sorted(v, key=lambda s: (v[s], s)); n = max(1, len(a))
    return {s: (i + 1.) / n for i, s in enumerate(a)}

def capped(raw, cap=.15):
    out = {s: 0.0 for s in UNIVERSE}; active = set(UNIVERSE); left = 1.0
    while active:
        den = sum(max(raw[s], 1e-8) for s in active)
        fixed = [s for s in active if left * max(raw[s], 1e-8) / den > cap]
        if not fixed:
            for s in active: out[s] = left * max(raw[s], 1e-8) / den
            break
        for s in fixed:
            out[s] = cap; left -= cap; active.remove(s)
    z = sum(out.values())
    return {s: out[s] / z for s in UNIVERSE}

@register_hook
def cross_asset_strategy():
    global WAIT
    if WAIT:
        WAIT -= 1; return
    prices, returns = {}, {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=280)
        if df is None: continue
        p = np.asarray(df.sort_values("date")["close"], dtype=float)[:-1]
        if len(p) >= 125 and np.all(np.isfinite(p)) and np.all(p > 0):
            prices[s] = p; returns[s] = p[1:] / p[:-1] - 1.0
    if len(prices) < 12:
        WAIT = 9; return
    def mom(s, n): return prices[s][-1] / prices[s][-n-1] - 1.0
    h10 = {s:mom(s,10) for s in prices}; h20 = {s:mom(s,20) for s in prices}
    h30 = {s:mom(s,30) for s in prices}; h40 = {s:mom(s,40) for s in prices}; h60 = {s:mom(s,60) for s in prices}
    v20 = {s:max(float(np.std(returns[s][-20:])), .008) for s in prices}; v40 = {s:max(float(np.std(returns[s][-40:])), .008) for s in prices}
    av = {n:float(np.mean([mom(s,n) for s in prices])) for n in (20,30,40,60)}
    stressed = av[20] < -.02 or float(np.mean([h60[s] > 0 for s in prices])) < .45
    defensive = {"XAU", "US10Y", "CN10Y"}; f = {}
    for s in prices:
        r=returns[s]; down=float(np.std(np.minimum(r[-30:],0.0))); path=abs(h20[s])/max(float(np.sum(abs(r[-20:]))),.001)
        f[s]=[-(h60[s]-av[60])/v40[s]+.15*(h20[s]-h60[s])/v20[s],
              -(h40[s]-av[40])/v40[s]+.25*(h10[s]-h40[s])/v20[s],
              path+.30*(h20[s]-av[20])/v20[s],
              (1. if s in defensive else 0.)+(h20[s]-av[20])/v20[s]-.2*max(0.,-h10[s])/v20[s],
              -(h30[s]-av[30])/v40[s]+.2*path,
              (1. if s in defensive else 0.)-max(0.,-h20[s])/v20[s],
              -down/v40[s]+.25*path,
              (h20[s]-av[20])/v20[s]+.25*(h10[s]-h20[s])/v20[s]]
    ranks=[rank01({s:f[s][j] for s in prices}) for j in range(8)]
    score={s:sum(FACTOR_WEIGHTS[j]*(ranks[j][s]-.5) for j in range(8)) for s in prices}
    tilt={s:1. for s in UNIVERSE}
    if stressed: tilt.update({"XAU":1.40,"US10Y":1.28,"CN10Y":1.20,"BTC":.68,"ETH":.65,"WTI":.82,"COPPER":.86})
    target=capped({s:tilt[s]*max(.51+score.get(s,0.),.05)/v20.get(s,.02)**.20 for s in UNIVERSE})
    account=get_account_dict(); assets=max(float(account.get("total_assets",1.)),1.)
    old={s:0. for s in UNIVERSE}
    for p in account.get("positions",[]):
        if p.get("symbol") in old and float(p.get("quantity",0))>0: old[p["symbol"]]=max(float(p.get("market_value",0)),0.)/assets
    if sum(old.values())>.001:
        target={s:.35*target[s]+.65*old[s] for s in UNIVERSE}; z=sum(target.values()); target={s:target[s]/z for s in UNIVERSE}
    forecast={s:float(.012*score.get(s,0.)*(.8 if stressed else 1.)) for s in UNIVERSE}
    rebalance_to_weights(target, forecast_returns=forecast, factor_ids=FACTOR_IDS, horizon_days=10)
    WAIT=9

__all__=["cross_asset_strategy"]
