from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acct = get_account_dict()


def loader(a):
    try:
        return get_stock_daily_data(a, days=120)
    except Exception:
        try:
            return get_index_daily_data(a, days=120)
        except Exception:
            return None


print("=== Match cost price to historical close date ===")
for p in acct.get("positions", []):
    a = p["symbol"]
    cost = p.get("cost_price", 0)
    df = loader(a)
    if df is None:
        print(f"  {a}: no data")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    diffs = (df["close"].astype(float) - cost).abs()
    i = diffs.idxmin()
    print(f"  {a:10s} cost={cost:.4f} best_match={df.iloc[i]['date'].date()} close={df.iloc[i]['close']:.4f} diff={diffs.iloc[i]:.6f}")

# Also check current weight per position
print("\n=== Current weights ===")
tot = acct.get("total_assets", 1)
for p in acct.get("positions", []):
    w = p.get("market_value", 0) / tot * 100
    print(f"  {p['symbol']:10s} w={w:6.2f}%")
