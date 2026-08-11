"""Trace _fit_weights iterations for the actual pref vector."""
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

n = len(assets)
z = [0.0] * n
for fac in ensemble:
    row = S._signal_row(fac["factor_id"], row_idx, n)
    zz = S._rank_z(row)
    z = [a + fac["weight"] * fac["direction"] * b for a, b in zip(z, zz)]
mean = sum(z) / n
var = sum((x - mean) ** 2 for x in z) / n
sd = math.sqrt(var) if var > 1e-14 else 1e-12
z_std = [(x - mean) / sd for x in z]

closes = S._closes(assets)
risk, vix, m20, disp = S._regime(closes, assets)
delta = 0.14 * risk
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

# replicate _fit_weights with tracing
total_pref = sum(max(0.0, float(x)) for x in pref.values())
w = {a: max(0.0, float(x)) / total_pref for a, x in pref.items()}
cap_a = {a: (cap_map.get(a, S.CAP) if cap_map else S.CAP) for a in w}
cap = S.CAP
floor = S.FLOOR
for it in range(500):
    excess = sum(max(0.0, w[a] - cap_a[a]) for a in w)
    if excess > 1e-12:
        room = [a for a in w if w[a] < cap_a[a] - 1e-12]
        if not room:
            print(f"iter {it}: no room, break")
            break
        den_r = sum(max(0.0, pref.get(a, 0.0)) for a in room)
        moved = 0.0
        for a in room:
            add = excess * (max(0.0, pref.get(a, 0.0)) / den_r if den_r > 1e-12 else 1.0 / len(room))
            add = min(add, cap_a[a] - w[a])
            if add > 1e-14:
                w[a] += add
                moved += add
        if moved <= 1e-12:
            print(f"iter {it}: no movement, break")
            break
        print(f"iter {it}: excess={excess:.6f} moved={moved:.6f}")
        print("   w:", {a: round(v, 4) for a, v in sorted(w.items(), key=lambda kv: -kv[1])})
    short = sum(max(0.0, floor - x) for x in w.values())
    if short > 1e-12:
        donors = [a for a in w if w[a] > floor + 1e-12]
        avail = sum(w[a] - floor for a in donors)
        if avail > 1e-12:
            for a in donors:
                w[a] -= short * (w[a] - floor) / avail
        for a in w:
            if w[a] < floor:
                w[a] = floor
        print(f"iter {it}: floor short={short:.6f} applied")
    if excess <= 1e-12 and short <= 1e-12:
        print(f"iter {it}: converged (excess={excess:.2e} short={short:.2e})")
        break
total = sum(w.values())
print("normalized final:", {a: round(v / total, 4) for a, v in sorted(w.items(), key=lambda kv: -kv[1])})
