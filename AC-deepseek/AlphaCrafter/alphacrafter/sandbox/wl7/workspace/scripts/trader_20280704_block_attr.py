"""Trader 2028-07-04 block attribution: compare account.json.bak (pre-step,
block start 06-20) vs account.json (block end 07-04). Quantities identical
=> no rebalance executed. Compute per-asset block return and PnL contribution."""
import json
from pathlib import Path

BASE = Path("../persistent")

def load(name):
    with open(BASE / name) as f:
        return json.load(f)

pre = load("account.json.bak")
post = load("account.json")

pre_pos = {p["symbol"]: p for p in pre["positions"]}
post_pos = {p["symbol"]: p for p in post["positions"]}

nav0 = pre["total_assets"]
nav1 = post["total_assets"]

print(f"NAV start (bak, 06-19 px): {nav0:,.2f}")
print(f"NAV end   (account, 07-04): {nav1:,.2f}")
print(f"Block PnL: {nav1 - nav0:+,.2f}  ({(nav1/nav0 - 1)*100:+.2f}%)")
print(f"Step-reported period return: -2.55% (vs NAV 1,014,761 at 06-20 open)")

# rebalance check
qty_diff = {s: post_pos[s]["quantity"] - pre_pos[s]["quantity"] for s in pre_pos}
moved = {s: q for s, q in qty_diff.items() if abs(q) > 1e-9}
print(f"\nQuantity changes (rebalance executed?): {len(moved)}  {moved if moved else 'NONE -> edge gate skip'}")

# per-asset block return & attribution (using bak prices as block start px)
contribs = []
for s in pre_pos:
    pp = pre_pos[s]
    qq = post_pos[s]
    p0 = pp["current_price"]
    p1 = post_pos[s]["current_price"]
    mv0 = pp["market_value"]
    ret = p1 / p0 - 1.0 if p0 else 0.0
    wt = mv0 / nav0
    contribs.append((s, wt, ret, wt * ret, p0, p1, qq["quantity"]))
contribs.sort(key=lambda x: x[3], reverse=True)
print(f"\n{'asset':10s} {'wt%':>6s} {'block ret%':>10s} {'contrib pp':>10s}  p0 -> p1")
tot = 0.0
for s, wt, ret, c, p0, p1, q in contribs:
    tot += c
    print(f"{s:10s} {wt*100:6.2f} {ret*100:10.2f} {c*100:10.2f}  {p0:.4f} -> {p1:.4f}")
print(f"\nSum contributions: {tot*100:+.2f}pp  vs actual {(nav1/nav0-1)*100:+.2f}%")

# defensive floor block contribution
floor = ["XAU", "US10Y", "CN10Y"]
f_contrib = sum(c for s, _, _, c, *_ in [(x[0], x[1], x[2], x[3]) for x in contribs] if s in floor)
f_wt = sum(x[1] for x in contribs if x[0] in floor)
print(f"\nDefensive floor {floor}: wt {f_wt*100:.1f}%, block contrib {f_contrib*100:+.2f}pp")

# frozen feeds
frozen = ["NDX", "SOX", "000688.SH", "CN10Y"]
fz_wt = sum(x[1] for x in contribs if x[0] in frozen)
print(f"Frozen feeds {frozen}: wt {fz_wt*100:.1f}% (dead weight, rets: " +
      ", ".join(f"{x[0]} {x[2]*100:+.1f}%" for x in contribs if x[0] in frozen) + ")")
