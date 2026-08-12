"""Trader post-block review for the 2028-10-23..11-06 live cycle.
Computes per-asset block returns (10-23 close -> 11-03 close), block PnL
contributions using held quantities (no execution occurred this block), and
block-level stats. Uses only data visible at the decision date."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def get(sym, n=40):
    try:
        return get_stock_daily_data(sym, days=n)
    except Exception:
        try:
            return get_index_daily_data(sym, days=n)
        except Exception:
            return None


def main():
    acc = json.load(open("../persistent/account.json"))
    pos = {p["symbol"]: p for p in acc["positions"]}
    start_net = 972643.81  # account.json.bak (block start, 10-23 close marking)
    end_net = acc["net_assets"]

    closes = {}
    for a in ASSETS:
        df = get(a)
        if df is None or len(df) < 20:
            print(f"{a}: NO DATA")
            continue
        df = df.sort_values("date").reset_index(drop=True)
        by_date = {str(d.date()): float(c) for d, c in zip(df["date"], df["close"])}
        closes[a] = by_date

    print(f"{'asset':10s} {'qty':>10s} {'w_11-03':>8s} {'ret_10-23..11-03':>16s} {'pnl_block':>10s}")
    total_pnl = 0.0
    for a in ASSETS:
        p = pos.get(a)
        if p is None or a not in closes:
            continue
        c = closes[a]
        p0 = c.get("2028-10-23")
        p1 = c.get("2028-11-03")
        if p0 is None or p1 is None:
            print(f"{a}: missing dates p0={p0} p1={p1}")
            continue
        qty = p["quantity"]
        ret = p1 / p0 - 1.0
        pnl = qty * (p1 - p0)
        total_pnl += pnl
        w = p["market_value"] / end_net
        print(f"{a:10s} {qty:10.4f} {w:8.3f} {ret:16.4f} {pnl:10.2f}")

    print(f"\nblock start net (bak): {start_net:,.2f}")
    print(f"block end   net      : {end_net:,.2f}")
    print(f"sum asset pnl (10-23..11-03): {total_pnl:,.2f}")
    print(f"block return: {(end_net/start_net - 1.0)*100:.3f}%")

    # aggregate stats
    winners = [(a, closes[a]["2028-11-03"]/closes[a]["2028-10-23"] - 1.0,
                pos[a]["quantity"]*(closes[a]["2028-11-03"]-closes[a]["2028-10-23"]))
               for a in ASSETS if a in pos and a in closes
               and "2028-10-23" in closes[a] and "2028-11-03" in closes[a]]
    winners.sort(key=lambda x: -x[2])
    print("\nTOP CONTRIBUTORS:")
    for a, r, pnl in winners[:5]:
        print(f"  {a:10s} ret={r:+.3f} pnl={pnl:+.2f}")
    print("WORST CONTRIBUTORS:")
    for a, r, pnl in winners[-6:]:
        print(f"  {a:10s} ret={r:+.3f} pnl={pnl:+.2f}")


if __name__ == "__main__":
    main()
