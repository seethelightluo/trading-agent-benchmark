"""Block attribution using simulator-implied prices (mv/qty) from bak vs current."""
import json

BAK = "../persistent/account.json.bak"
CUR = "../persistent/account.json"
bak = json.load(open(BAK))
cur = json.load(open(CUR))

def posmap(a):
    return {p["symbol"]: p for p in a.get("positions", [])}

bp, cp = posmap(bak), posmap(cur)
na_start, na_end = bak["net_assets"], cur["net_assets"]
rows = []
for sym, p in bp.items():
    q = p["quantity"]
    p0 = p["market_value"] / q
    p1 = cp[sym]["market_value"] / q
    ret = p1 / p0 - 1.0
    wt = p["market_value"] / na_start
    rows.append((sym, wt, ret, wt * ret, p["market_value"]))

rows.sort(key=lambda r: -abs(r[3]))
print(f"start {na_start:.2f} end {na_end:.2f} actual ret {(na_end/na_start-1)*100:.3f}%")
print(f"{'sym':9s} {'wt%':>6s} {'ret%':>8s} {'contrib_pp':>9s}")
tot = 0.0
for sym, wt, ret, contrib, mv in rows:
    print(f"{sym:9s} {wt*100:6.2f} {ret*100:8.2f} {contrib*100:9.2f}")
    tot += contrib
print(f"\nsum contrib {tot*100:.3f} pp")
