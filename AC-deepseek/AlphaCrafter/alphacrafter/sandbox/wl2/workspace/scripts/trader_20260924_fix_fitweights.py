"""Trader fix 2026-09-24 (v2): replace broken _fit_weights in strategy.py.

Old implementation never reduced capped assets during excess redistribution;
it inflated total weight and renormalized, flattening targets to ~equal weight.
Corrected iterative cap/floor fit below.
"""
import ast
import random
from pathlib import Path

BASE = Path(__file__).parent.parent
path = BASE / "strategy.py"
src = path.read_text()

NEW_FIT = '''def _fit_weights(pref, cap=CAP, floor=FLOOR, cap_map=None):
    """Iterative cap/floor normalization of a non-negative preference vector.

    cap_map optionally overrides the cap per asset (e.g., trend-failing assets).
    Preserves sum-to-1: excess above caps is taken from capped assets and
    redistributed to assets below cap (proportional to pref); assets below
    floor are raised by taking from donors above floor. Fixed 2026-09-24:
    the previous implementation never reduced capped assets, so it inflated
    the total and renormalized everything to ~equal weight.
    """
    total_pref = sum(max(0.0, float(x)) for x in pref.values())
    if total_pref <= 0.0:
        n = len(pref)
        return {a: 1.0 / n for a in pref}
    w = {a: max(0.0, float(x)) / total_pref for a, x in pref.items()}
    cap_a = {a: (cap_map.get(a, cap) if cap_map else cap) for a in w}
    n = len(w)
    for _ in range(200):
        excess = 0.0
        for a in w:
            if w[a] > cap_a[a]:
                excess += w[a] - cap_a[a]
                w[a] = cap_a[a]
        short = 0.0
        for a in w:
            if w[a] < floor:
                short += floor - w[a]
                w[a] = floor
        s = sum(w.values())
        need = 1.0 - s
        if need > 1e-12:
            room = [a for a in w if w[a] < cap_a[a] - 1e-12]
            remaining = need
            while remaining > 1e-12 and room:
                den = sum(max(0.0, pref.get(a, 0.0)) for a in room)
                if den <= 1e-12:
                    given = 0.0
                    for a in room:
                        add = min(remaining / len(room), cap_a[a] - w[a])
                        w[a] += add
                        given += add
                    remaining -= given
                    break
                given = 0.0
                for a in room:
                    share = remaining * pref.get(a, 0.0) / den
                    add = min(share, cap_a[a] - w[a])
                    if add > 1e-14:
                        w[a] += add
                        given += add
                remaining -= given
                room = [a for a in room if w[a] < cap_a[a] - 1e-12]
                if given <= 1e-12:
                    break
            if remaining > 1e-12:
                break
        elif need < -1e-12:
            donors = [a for a in w if w[a] > floor + 1e-12]
            remaining = -need
            while remaining > 1e-12 and donors:
                avail = sum(w[a] - floor for a in donors)
                if avail <= 1e-12:
                    break
                removed = 0.0
                for a in donors:
                    share = remaining * (w[a] - floor) / avail
                    cut = min(share, w[a] - floor)
                    if cut > 1e-14:
                        w[a] -= cut
                        removed += cut
                remaining -= removed
                donors = [a for a in w if w[a] > floor + 1e-12]
                if removed <= 1e-12:
                    break
            if remaining > 1e-12:
                break
        if excess <= 1e-12 and short <= 1e-12 and abs(need) <= 1e-12:
            break
    total = sum(w.values())
    if total <= 0.0:
        return {a: 1.0 / n for a in w}
    return {a: x / total for a, x in w.items()}'''

start = src.index("def _fit_weights(")
end = src.index("def build_target(")
src = src[:start] + NEW_FIT + "\n\n\n" + src[end:]
path.write_text(src)
ast.parse(src)
print("patched OK, parse OK")

# ---- stress test ----
import strategy

def check(pref, cap, floor, cap_map=None, label=""):
    w = strategy._fit_weights(pref, cap=cap, floor=floor, cap_map=cap_map)
    tot = sum(w.values())
    ok = abs(tot - 1.0) < 1e-6
    cap_ok = all(v <= cap_a + 1e-6 for v, cap_a in zip(w.values(), (cap_map or {}).values()))
    floor_ok = all(v >= floor - 1e-6 for v in w.values())
    print(f"{label:34s} sum={tot:.4f} ok={ok} floor_ok={floor_ok}")
    return ok

check({'A': 0.9, 'B': 0.05, 'C': 0.05}, 0.5, 0.01, label="synthetic cap")
check({'A': 0.8, 'B': 0.1, 'C': 0.1}, 0.17, 0.012, {'A': 0.09}, label="cap_map feasible-ish")
check({'A': 0.01, 'B': 0.99}, 0.8, 0.4, label="floor lift")

random.seed(7)
all_ok = True
for t in range(200):
    n = 15
    pref = {f"A{i}": random.random() ** 2 for i in range(n)}
    cap_map = {f"A{i}": 0.09 if random.random() < 0.2 else 0.17 for i in range(n)}
    w = strategy._fit_weights(pref, cap=0.17, floor=0.012, cap_map=cap_map)
    tot = sum(w.values())
    if abs(tot - 1.0) > 1e-6:
        all_ok = False
        print("FAIL sum", t, tot)
        break
    for a, v in w.items():
        if v < 0.012 - 1e-6 or v > cap_map[a] + 1e-6:
            all_ok = False
            print("FAIL bound", t, a, v, cap_map[a])
            break
    if not all_ok:
        break
print("stress 200 trials:", "ALL OK" if all_ok else "FAILED")
