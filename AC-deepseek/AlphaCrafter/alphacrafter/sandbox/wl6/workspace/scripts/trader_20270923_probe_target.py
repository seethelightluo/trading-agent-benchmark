import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import strategy as S
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
assets = acc.get("watch_list", [])

panel = {}
for a in assets:
    df = S.stock(a) if a in ("000300.SH", "000688.SH") else S.index(a)
    if df is not None and len(df) > 0:
        panel[a] = df
closes = {a: df["close"].values for a, df in panel.items()}
vix_df = S.index("VIX")
vix_close = vix_df["close"].values if vix_df is not None else None
factors = S.load_ensemble()
raw = S.compute_raw_factors(closes, vix_close, assets)
score = {}
for a in assets:
    s = 0.0
    for f in factors:
        v = raw[f["factor_id"]].get(a)
        if v is None:
            v = 0.0
        s += f["weight"] * v * f["direction"]
    score[a] = s

sarr = np.array([score[a] for a in assets])
regime = S.regime_from_market(panel)
K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
print("regime:", regime, "K:", K, "n_factors:", len(factors))
top = np.argsort(-sarr)[:K]
w = np.zeros(len(assets))
w[top] = 1.0 / K
vol = S.vol20_map(closes, assets)
v = np.array([vol[a] if vol.get(a) else np.nan for a in assets])
vinv = np.where(np.isfinite(v) & (v > 0), 1.0 / np.maximum(v, 1e-9), 0.0)
vinv[~np.isfinite(v)] = 0.0
if vinv.sum() > 0:
    w = 0.6 * w + 0.4 * (vinv / vinv.sum())
w = S.apply_floor(w, assets, [a for a in S.DEF if a in assets], S.FLOOR[regime])
w = S.apply_cap(w, assets)
w = S.apply_min_xau(w, assets)
w = S.apply_crypto_cap(w, assets)
w = w / w.sum()

print("TARGET (as of latest data):")
for a, wi in sorted(zip(assets, w), key=lambda x: -x[1]):
    print("  %-10s %6.2f%%" % (a, wi * 100))
print("sum:", round(w.sum(), 6))

mv = {p["symbol"]: p["market_value"] for p in acc.get("positions", [])}
tot = sum(mv.values())
print("ACTUAL current weights:")
for a in sorted(assets, key=lambda x: -mv.get(x, 0)):
    print("  %-10s %6.2f%%" % (a, mv.get(a, 0) / tot * 100))
