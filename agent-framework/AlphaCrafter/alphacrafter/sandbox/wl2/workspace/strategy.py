import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener ensemble, capped at ten active factors; all directions are positive.
FW = {"compression": .16, "stress": .14, "breadth": .13, "quiet": .12,
      "relative": .11, "risk_rev": .10, "dispersion_rev": .08,
      "downside": .07, "agreement_rev": .05, "volume_rev": .04}
CADENCE = 10
_day = 0

def cs_rank(x):
    good = sorted((s, float(v)) for s, v in x.items() if np.isfinite(v))
    out = {s: .5 for s in ASSETS}
    if len(good) > 1:
        for i, (s, _) in enumerate(sorted(good, key=lambda z: z[1])):
            out[s] = i / (len(good) - 1)
    return out

def weights(score):
    # Full investment, 2% floor and 20% cap; redistribution is deterministic.
    floor, cap = .02, .20
    x = {s: max(float(score.get(s, .5)), .01) for s in ASSETS}
    w = {s: floor + (1 - floor*len(ASSETS))*x[s]/sum(x.values()) for s in ASSETS}
    for _ in range(30):
        hi = [s for s in ASSETS if w[s] > cap]
        if not hi: break
        excess = sum(w[s]-cap for s in hi)
        for s in hi: w[s] = cap
        lo = [s for s in ASSETS if s not in hi]
        z = sum(x[s] for s in lo)
        for s in lo: w[s] += excess*x[s]/max(z, 1e-12)
    z = sum(w.values())
    return {s: max(0.0, w[s]/z) for s in ASSETS}

@register_hook
def cross_asset_strategy():
    global _day
    _day += 1
    if (_day - 1) % CADENCE: return
    feat = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=280)
        if df is None or len(df) < 100: continue
        c = np.asarray(df.sort_values("date").iloc[:-1].close, dtype=float)
        c = c[np.isfinite(c) & (c > 0)]
        if len(c) < 80: continue
        r = c[1:]/c[:-1]-1
        neg = r[-30:][r[-30:] < 0]
        feat[s] = dict(t3=c[-1]/c[-4]-1, t10=c[-1]/c[-11]-1,
          t20=c[-1]/c[-21]-1, t30=c[-1]/c[-31]-1, t60=c[-1]/c[-61]-1,
          v10=max(np.std(r[-10:]), .003), v20=max(np.std(r[-20:]), .003),
          v60=max(np.std(r[-60:]), .003), downv=max(np.std(neg) if len(neg)>2 else np.std(r[-20:]), .003),
          activity=np.mean(np.abs(r[-3:]))/max(np.mean(np.abs(r[-20:])), .003))
    if len(feat) < 10: return
    m3=np.mean([d['t3'] for d in feat.values()]); m60=np.mean([d['t60'] for d in feat.values()])
    disp=max(np.std([d['t10'] for d in feat.values()]), .01)
    raw={}
    for s,d in feat.items():
        agree=np.mean([d['t10']>0,d['t20']>0,d['t60']>0]); rev=-(d['t3']-m3)/(d['v20']+.01)
        raw[s] = {
          'compression': -d['v10']/(d['v60']+.003),
          'stress': (d['t10'] if d['t30']>=0 else -d['t10'])/(d['v20']+.01),
          'breadth': d['t30']/(d['v60']+.01), 'quiet': d['t20']*abs(d['t20'])/(d['v60']+.01),
          'relative': d['t20']/(d['v20']+.01), 'risk_rev': -(d['t60']-m60)/(d['v60']+.01),
          'dispersion_rev': rev*min(2., disp/.03), 'downside': d['t60']/(d['downv']+.01),
          'agreement_rev': rev*(.5+.5*agree), 'volume_rev': rev*(1+.2*min(d['activity'],2.))}
    ranks={f:cs_rank({s:raw[s][f] for s in raw}) for f in FW}
    score={s:sum(FW[f]*ranks[f].get(s,.5) for f in FW) for s in ASSETS}
    breadth=np.mean([d['t30']>0 for d in feat.values()]); spx=feat.get('SPX',{}).get('t30',0.)
    risk=min(1., max(0.,(.50-breadth)*1.8)+max(0.,(-spx-.01)/.15))
    for s in ('XAU','US10Y','CN10Y'): score[s] *= 1+1.25*risk
    for s in ('BTC','ETH','WTI','SOX','NDX','COPPER'): score[s] *= max(.15,1-.75*risk)
    med=np.median([d['v20'] for d in feat.values()])
    for s in ASSETS: score[s]=max(.01,score[s])*np.clip(med/max(feat.get(s,{}).get('v20',med),.003),.75,1.20)
    target=weights(score)
    if set(target)==set(ASSETS) and abs(sum(target.values())-1)<1e-8:
        rebalance_to_weights(target)

def strategy():
    cross_asset_strategy()
