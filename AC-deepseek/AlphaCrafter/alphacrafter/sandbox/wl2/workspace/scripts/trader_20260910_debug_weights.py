"""Debug: trace the weight construction path for 2026-09-10."""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import strategy as S
from alphacrafter.sim.utils import get_account_dict

date_state = json.loads(Path("../persistent/date.json").read_text())
account = get_account_dict()
assets = list(account.get("watch_list", []))
ensemble = S._load_ensemble()

trading_days = date_state["trading_days"]
visible = date_state["visible_through"]
row_idx = trading_days.index(visible) - trading_days.index(S.ARTIFACT_START)
print("row_idx", row_idx, "visible", visible)

n = len(assets)
z = [0.0] * n
used = []
for fac in ensemble:
    row = S._signal_row(fac["factor_id"], row_idx, n)
    zz = S._rank_z(row)
    z = [a + fac["weight"] * fac["direction"] * b for a, b in zip(z, zz)]
    used.append(fac["factor_id"])

mean = sum(z) / n
var = sum((x - mean) ** 2 for x in z) / n
sd = math.sqrt(var) if var > 1e-14 else 1e-12
z_std = [(x - mean) / sd for x in z]
print("z_std:", dict(zip(assets, [round(x, 3) for x in z_std])))

closes = S._closes(assets)
risk, vix, m20, disp = S._regime(closes, assets)
delta = 0.14 * risk
print("risk", round(risk, 3), "delta", round(delta, 4))

mx = max(z_std)
exps = [math.exp(x - mx) for x in z_std]
den = sum(exps)
base = {a: exps[i] / den for i, a in enumerate(assets)}
print("base softmax:", {a: round(v, 4) for a, v in sorted(base.items(), key=lambda kv: -kv[1])})

pref = {}
for i, a in enumerate(assets):
    if a in S.DEFENSIVE:
        pref[a] = base[a] + delta / len(S.DEFENSIVE)
    else:
        pref[a] = base[a] * (1.0 - delta)
print("pref:", {a: round(v, 4) for a, v in sorted(pref.items(), key=lambda kv: -kv[1])})
print("pref sum:", round(sum(pref.values()), 6))

r20 = {}
for a in assets:
    c = closes.get(a)
    r20[a] = float(c.iloc[-1] / c.iloc[-21] - 1.0) if (c is not None and len(c) >= 21) else 0.0
cap_map = {a: S.TREND_CAP for a in assets if r20[a] < S.TREND_THRESH}
print("cap_map:", cap_map)
weights = S._fit_weights(pref, cap=S.CAP, floor=S.FLOOR, cap_map=cap_map or None)
print("final weights:", {a: round(v, 4) for a, v in sorted(weights.items(), key=lambda kv: -kv[1])})
print("sum:", round(sum(weights.values()), 6))
