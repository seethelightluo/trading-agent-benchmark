"""miner_2 2030-09-19: Family A - momentum freshness / parabolic surge guard.
Motivation: trader memory flags ~35 consecutive stale-ensemble cycles; #1 recurring drag is
commodity/crypto momentum-add whipsaw (WTI/COPPER/ETH added after parabolic 10-20d surges
then reversed). Idea: a momentum signal that penalizes 'fresh parabolic' moves and rewards
'sustained' trends. Variants tested at 10d admission horizon."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from miner2_20300919_lib import ASSETS, load_prices, load_macro, eval_factor, library_signals

px, ret = load_prices()
print("price matrix:", px.shape, "dates:", px.index[0].date(), "->", px.index[-1].date())

# --- candidate constructions ---
mom20 = px.pct_change(20).shift(5)            # 20d momentum, skip 5 (library convention)
ret5 = px.pct_change(5)                       # raw recent 5d
ret10 = px.pct_change(10)                     # raw recent 10d
ret20 = px.pct_change(20)                     # raw recent 20d
mom60 = px.pct_change(60).shift(5)

# A1: sustained momentum = mom20 * (1 - |ret5|/|ret20|)  (ratio -> 1 when move is all recent = parabolic)
ratio_5_20 = ret5.abs() / ret20.abs().replace(0, np.nan)
a1 = mom20 * (1 - ratio_5_20.clip(upper=1.0))
# A2: parabolic guard (pure penalty on fresh surge, direction-agnostic damp of |mom|)
a2 = mom20 * (1 - (ret10.abs() / ret20.abs().replace(0, np.nan)).clip(upper=1.0))
# A3: surge penalty only when momentum positive (chasing risk), no penalty when negative
a3 = mom20 * np.where(mom20 > 0, (1 - ratio_5_20.clip(upper=1.0)), 1.0)
# A4: momentum damped by 10d realized vol rank (freshness via vol spike)
a4 = mom20 / (1 + ret.rolling(10).std())
# A5: efficiency ratio (Kaufman) x momentum: trend efficiency
def eff_ratio(s, n=20):
    change = s.diff(n).abs()
    vol = s.diff().abs().rolling(n).sum()
    return (change / vol).replace([np.inf, -np.inf], np.nan)
er20 = eff_ratio(px)
a5 = mom20 * er20.shift(5)

cands = {
    "sustained_mom_20": a1,
    "fresh_guard_20": a2,
    "chase_guard_20": a3,
    "vol_damped_mom_20": a4,
    "eff_mom_20": a5,
}

print("\n=== Family A validation (horizon 10, skip 0) ===")
for name, f in cands.items():
    m = eval_factor(f, ret, horizon=10, skip=0, name=name)
    print(m)

print("\n=== Decay by horizon for best candidates ===")
for name in ["sustained_mom_20", "chase_guard_20", "eff_mom_20"]:
    f = cands[name]
    row = {"name": name}
    for h in [1, 2, 3, 5, 10, 20]:
        m = eval_factor(f, ret, horizon=h, skip=0, name=name)
        row[f"h{h}"] = m["ic"]
    print(row)

print("\n=== Sub-period robustness (horizon 10) ===")
for name in ["sustained_mom_20", "chase_guard_20", "eff_mom_20"]:
    f = cands[name]
    fwd = ret.shift(-10).rolling(10).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    print("---", name)
    for lab, lo, hi in [("2020-2022", "2020-01-01", "2022-12-31"), ("2023-2025", "2023-01-01", "2025-12-31"),
                        ("2026+", "2026-01-01", "2030-12-31"), ("last250", None, None)]:
        fsub = f.loc[lo:hi] if lo else f.iloc[-250:]
        fwsub = fwd.loc[lo:hi] if lo else fwd.iloc[-250:]
        ics = []
        for d in fsub.index.intersection(fwsub.index):
            ff, rr = fsub.loc[d], fwsub.loc[d]
            m = ff.notna() & rr.notna()
            if m.sum() >= 8:
                ics.append(ff[m].rank().corr(rr[m].rank()))
        ics = pd.Series(ics).dropna()
        if len(ics) >= 30:
            print(f"  {lab}: n={len(ics)} ic={ics.mean():.4f} icir={ics.mean()/ics.std(ddof=1)*np.sqrt(len(ics)):.3f}")

print("\n=== Library correlation (ranked signal, full overlap) ===")
lib = library_signals(px, ret)
for name in ["sustained_mom_20", "chase_guard_20"]:
    f = cands[name].rank(axis=1)
    print("---", name)
    for k, v in lib.items():
        both = f.join(v, lsuffix="_x", rsuffix="_y").dropna()
        if len(both) > 0:
            rho = both.iloc[:, 0].corr(both.iloc[:, 1])
            print(f"  {k}: {rho:+.3f}")
