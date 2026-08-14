from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acct = get_account_dict()
START_TOTAL = 1216697.17  # account value at start of 06-11 block (per v63 memory)


def loader(a):
    try:
        return get_stock_daily_data(a, days=120)
    except Exception:
        try:
            return get_index_daily_data(a, days=120)
        except Exception:
            return None


print("=== 06-11..06-25 block decomposition (holdings from 05-28 execution) ===")
tot_contrib = 0.0
for p in acct.get("positions", []):
    a = p["symbol"]
    qty = p.get("quantity", 0)
    df = loader(a)
    if df is None:
        continue
    df = df.sort_values("date").reset_index(drop=True)
    # 06-11 close
    m11 = df["date"] == pd.Timestamp("2035-06-11")
    m22 = df["date"] == pd.Timestamp("2035-06-22")
    if not m11.any() or not m22.any():
        # fallback: first row >= 06-11 and last row
        c11 = df.loc[df["date"] >= pd.Timestamp("2035-06-11"), "close"].iloc[0]
        c22 = df.iloc[-1]["close"]
    else:
        c11 = df.loc[m11, "close"].iloc[0]
        c22 = df.loc[m22, "close"].iloc[0]
    ret = float(c22) / float(c11) - 1.0
    mv_start = qty * float(c11)
    w = mv_start / START_TOTAL
    contrib = w * ret * 100
    tot_contrib += contrib
    print(f"  {a:10s} ret={ret*100:>8.3f}%  w06-11={w*100:6.2f}%  contrib={contrib:>7.3f}%")

print(f"\n  SUM contrib = {tot_contrib:.3f}%  (actual block = +1.085%)")
