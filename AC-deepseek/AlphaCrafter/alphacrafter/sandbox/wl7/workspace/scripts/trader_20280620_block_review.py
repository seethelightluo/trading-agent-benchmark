"""Trader block review: 2028-06-06 -> 2028-06-20.
Compare account.json vs account.json.bak (pre-step) to detect rebalance execution,
compute per-asset block PnL contribution and closing prices.
Read-only inspection; does not advance the account.
"""
import json, glob, os
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "persistent")

def load_account(path):
    with open(path) as f:
        return json.load(f)

pre = load_account(os.path.join(BASE, "account.json.bak"))
post = load_account(os.path.join(BASE, "account.json"))

def pos_map(acc):
    return {p["symbol"]: p for p in acc.get("positions", [])}

pre_p, post_p = pos_map(pre), pos_map(post)

print("=== ACCOUNT ===")
for k in ("total_assets", "net_assets", "available_cash", "market_value", "gross_position_rate", "net_position_rate"):
    print(f"  pre  {k:22s}: {pre.get(k)}")
    print(f"  post {k:22s}: {post.get(k)}")

print("\n=== POSITION CHANGES (qty pre -> post) ===")
changed = False
for sym in sorted(set(pre_p) | set(post_p)):
    q0 = pre_p.get(sym, {}).get("quantity", 0)
    q1 = post_p.get(sym, {}).get("quantity", 0)
    if abs(q0 - q1) > 1e-9:
        changed = True
        print(f"  {sym:10s} {q0:.4f} -> {q1:.4f}  (CHANGED)")
if not changed:
    print("  NO QUANTITY CHANGES -> no rebalance executed (edge gate skip)")

print("\n=== BLOCK ATTRIBUTION (close-based) ===")
# use stock_data dir for closes
sdir = os.path.join(BASE, "stock_data")
rows = []
for sym in sorted(pre_p):
    p0 = pre_p.get(sym)
    if p0 is None or p0.get("quantity", 0) <= 0:
        continue
    qty = p0["quantity"]
    # find last close before block start (2028-06-06) and last close at block end
    fpath = None
    for cand in glob.glob(os.path.join(sdir, "*" + sym + "*")):
        fpath = cand
    if fpath is None:
        continue
    df = pd.read_csv(fpath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # pre-step backup reflects prices at 2028-06-06 (block start after prev close 06-05)
    # Use cost basis -> current plr from pre account as start anchor is complex;
    # simpler: compute return over block using closes around 06-05 and 06-19
    start_mask = df["date"] <= pd.Timestamp("2028-06-05")
    end_mask = df["date"] <= pd.Timestamp("2028-06-19")
    if start_mask.sum() == 0 or end_mask.sum() == 0:
        continue
    c0 = df.loc[start_mask, "close"].iloc[-1]
    c1 = df.loc[end_mask, "close"].iloc[-1]
    ret = c1 / c0 - 1.0
    mv0 = qty * c0
    contrib = mv0 * ret
    rows.append((sym, qty, c0, c1, ret, contrib))

tot = sum(r[5] for r in rows)
for sym, qty, c0, c1, ret, contrib in sorted(rows, key=lambda r: -r[5]):
    print(f"  {sym:10s} qty {qty:10.2f}  {c0:10.2f}->{c1:10.2f}  {ret*100:+7.2f}%  contrib {contrib:12.0f} ({contrib/tot*100 if tot else 0:+6.2f}pp)")
print(f"  TOTAL start MV {tot:.0f}")

print("\n=== PENDING ORDERS ===")
print("  pre:", len(pre.get("orders", [])))
print("  post:", len(post.get("orders", [])))
