"""Debug build_target internals for WTI cap."""
import json, sys, math
sys.path.insert(0, ".")
from pathlib import Path
from alphacrafter.sim.utils import get_account_dict
import strategy as S

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
date_state = json.loads(Path("../persistent/date.json").read_text())
cur_w = S._current_weights(acc, assets)
ensemble = S._load_ensemble()

# replicate build_target internals step by step
n = len(assets)
live = S._live_factors(assets)
z = [0.0] * n
used = []
for fac in ensemble:
    fid = fac["factor_id"]
    lv = live.get(fid)
    row = lv if (lv is not None and sum(1 for v in lv if v == v) >= S.LIVE_MIN_FINITE) else S._signal_row(fid, 0, n)
    if row is None:
        continue
    zz = S._rank_z(row)
    z = [a + fac["weight"] * fac["direction"] * b for a, b in zip(z, zz)]
    used.append(fid)
print("used:", used)
mean = sum(z) / n
var = sum((x - mean) ** 2 for x in z) / n
sd = math.sqrt(var) if var > 1e-14 else 1e-12
z_std = [(x - mean) / sd for x in z]

closes = S._closes(assets)
risk, vix, m20, disp = S._regime(closes, assets)
delta = 0.16 * risk
mx = max(z_std)
exps = [math.exp(x - mx) for x in z_std]
den = sum(exps)
base = {a: exps[i] / den for i, a in enumerate(assets)}
pref = {}
for i, a in enumerate(assets):
    if a in S.DEFENSIVE:
        pref[a] = base[a] + delta / len(S.DEFENSIVE)
    else:
        pref[a] = base[a] * (1.0 - delta)

r20 = {}
for a in assets:
    c = closes.get(a)
    r20[a] = float(c.iloc[-1] / c.iloc[-21] - 1.0) if (c is not None and len(c) >= 21) else 0.0
cap_map = {a: S.TREND_CAP for a in assets if r20[a] < S.TREND_THRESH}
for a in S.CHINA:
    if r20[a] < S.CHINA_TREND_THRESH:
        cap_map[a] = min(cap_map.get(a, S.CAP), S.CHINA_TREND_CAP)
for a in S.MOMENTUM_ADD:
    if r20[a] < S.MOM_TREND_THRESH or vix >= S.MOM_VIX_THRESH:
        cap_map[a] = min(cap_map.get(a, S.CAP), S.MOM_TREND_CAP)
for a in S.COMMODITY_PARABOLIC:
    if r20[a] > S.COMMODITY_PARABOLIC_THRESH:
        cap_map[a] = min(cap_map.get(a, S.CAP), S.COMMODITY_PARABOLIC_CAP)
print("r20 WTI:", r20["WTI"])
print("cap_map:", cap_map)
weights = S._fit_weights(pref, cap=S.CAP, floor=S.FLOOR, cap_map=cap_map or None)
print("WTI weight after fit:", weights["WTI"], "SPX:", weights["SPX"], "COPPER:", weights["COPPER"])
turn = sum(abs(weights[a] - max(0.0, cur_w.get(a, 0.0))) for a in assets)
print("turn before dampener:", turn)
if turn > S.MAX_TURNOVER:
    s = S.MAX_TURNOVER / turn
    weights = {a: max(0.0, cur_w.get(a, 0.0)) + s * (weights[a] - max(0.0, cur_w.get(a, 0.0))) for a in assets}
    tot = sum(weights.values())
    if tot > 0:
        weights = {a: w / tot for a, w in weights.items()}
print("WTI final:", weights["WTI"])
