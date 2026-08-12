"""Trader inspection at 2030-07-05: compute current proposal + regime state."""
import json, sys
sys.path.insert(0, ".")
import strategy as S

cur, tds = S._today_and_calendar()
print("current_date:", cur, "| last_proposal:", S._last_proposal_date(tds))
print("should_propose:", S._should_propose(cur, tds), "| is_rebalance_day:", S._is_rebalance_day(cur, tds))

acc = S.get_account_dict()
assets = list(acc.get("watch_list", []))
print("n_assets:", len(assets))
print("cash:", acc.get("available_cash"), "gross_pos_rate:", acc.get("gross_position_rate"),
      "net_assets:", acc.get("net_assets"))

frames = S._fetch(assets)
scores, used = S._scores(frames, assets, cur)
print("factors_used:", used, "| FACTORS:", [(f, w, d) for f, w, d in S.FACTORS])

regime = S._regime(frames, assets)
below = S._below_ma(frames, assets)
print("regime:", regime)
print("below_MA20:", sorted(below))

# factor value table
for fid, w, d in S.FACTORS:
    vals = S._factor_values(frames, fid, cur)
    r = S._ranks(vals, assets)
    order = sorted(assets, key=lambda a: (r[a], a))
    top = [(a, round(vals[a], 4)) for a in order[-4:]]
    bot = [(a, round(vals[a], 4)) for a in order[:3]]
    print(f"  {fid} w={w} d={d} top={top} bot={bot}")

sc = sorted(assets, key=lambda a: (scores[a], a))
print("composite top6:", [(a, round(scores[a], 4)) for a in sc[-6:]])
print("composite bot3:", [(a, round(scores[a], 4)) for a in sc[:3]])

scores2 = S._de_rank_value_traps(dict(scores), frames, assets, cur)
w = S._weights(scores2, assets, regime)
w = S._composite_top2_cap(w, assets, scores2)
w = S._composite_ma_guard(w, frames, assets)
w = S._ma_guard(w, frames, assets, cur)
w = S._crypto_cap(w, assets)
w = S._commod_cap(w, assets)

print("\n--- proposed weights (current guard order v9->v11) ---")
for a in sorted(w, key=lambda x: -w[x]):
    print(f"  {a:8s} {w[a]*100:6.2f}%")

csum = w.get("BTC", 0) + w.get("ETH", 0)
comsum = w.get("WTI", 0) + w.get("COPPER", 0)
print(f"crypto sum: {csum*100:.2f}% (cap 12) | commod sum: {comsum*100:.2f}% (cap 14)")
print("sum:", sum(w.values()))

# swapped order
w2 = S._weights(scores2, assets, regime)
w2 = S._composite_top2_cap(w2, assets, scores2)
w2 = S._composite_ma_guard(w2, frames, assets)
w2 = S._ma_guard(w2, frames, assets, cur)
w2 = S._commod_cap(w2, assets)
w2 = S._crypto_cap(w2, assets)
csum2 = w2.get("BTC", 0) + w2.get("ETH", 0)
comsum2 = w2.get("WTI", 0) + w2.get("COPPER", 0)
print("\n--- swapped order (v11->v9) ---")
print(f"crypto sum: {csum2*100:.2f}% (cap 12) | commod sum: {comsum2*100:.2f}% (cap 14)")
for a in sorted(w2, key=lambda x: -w2[x]):
    print(f"  {a:8s} {w2[a]*100:6.2f}%")

# crypto momentum/MA state
for a in ["BTC", "ETH"]:
    df = frames.get(a)
    if df is not None:
        c = float(df["close"].iloc[-1])
        ma20 = float(df["close"].rolling(20).mean().iloc[-1])
        mom20 = float(df["close"].iloc[-1] / df["close"].iloc[-21] - 1) if len(df) >= 21 else float("nan")
        mom120 = float(df["close"].shift(5).iloc[-1] / df["close"].shift(125).iloc[-1] - 1) if len(df) >= 126 else float("nan")
        print(f"{a}: close={c:.2f} ma20={ma20:.2f} below={c < ma20} mom20={mom20*100:.2f}% mom120={mom120*100:.2f}%")

f = S._forecasts(scores2, assets)
print("\nforecast sample:", {a: round(f[a], 4) for a in sc[-3:]}, {a: round(f[a], 4) for a in sc[:3]})
