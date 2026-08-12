"""Compute 10d-block returns for the missed windows 12-02..12-16 and 12-16..12-30
plus this block 12-30..01-13, to evaluate consecutive-streak watches for v37."""
import json
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = get_account_dict()["watch_list"]

# trading days in the relevant window (from date.json)
days = json.load(open("../persistent/date.json"))["trading_days"]
recent = [d for d in days if "2030-11-14" <= d <= "2031-01-14"]
# anchor closes: for each anchor date, find close of that trading day
anchors = ["2030-12-02", "2030-12-16", "2030-12-30", "2031-01-13"]
print("trading days:", recent[:3], "...", recent[-3:], "n=", len(recent))


def get(a, n=60):
    try:
        return get_stock_daily_data(a, days=n)
    except Exception:
        try:
            return get_index_daily_data(a, days=n)
        except Exception:
            return None


def close_on(df, date_str):
    d = df[df["date"].astype(str).str[:10] == date_str]
    return float(d["close"].iloc[0]) if len(d) else None


blocks = [("12-02..12-16", "2030-12-02", "2030-12-16"),
          ("12-16..12-30", "2030-12-16", "2030-12-30"),
          ("12-30..01-13", "2030-12-30", "2031-01-13")]

print(f"\n{'asset':10s}", end="")
for name, _, _ in blocks:
    print(f" {name:>12s}", end="")
print()

block_ret = {}
for a in assets:
    df = get(a)
    if df is None:
        continue
    df = df.sort_values("date")
    block_ret[a] = {}
    print(f"{a:10s}", end="")
    for name, d0, d1 in blocks:
        p0 = close_on(df, d0)
        p1 = close_on(df, d1)
        if p0 and p1:
            r = (p1 / p0 - 1.0) * 100.0
            block_ret[a][name] = r
            print(f" {r:11.2f}%", end="")
        else:
            print(f" {'n/a':>12s}", end="")
    print()

print("\n--- v36 watch evaluation (consecutive streaks) ---")
for a in assets:
    r = block_ret.get(a, {})
    seq = [r.get(b) for b in ("12-02..12-16", "12-16..12-30", "12-30..01-13")]
    if all(v is not None for v in seq):
        print(f"{a:10s} {' / '.join(f'{v:+.2f}' for v in seq)}")
