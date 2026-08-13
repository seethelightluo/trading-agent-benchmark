"""Replicate the 2033-03-11 decision (data through 03-10) to verify guard stack."""
import json
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import strategy as st

CUR = "2033-03-11"

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
assert len(assets) == 15, len(assets)

# fetch raw frames (through current 03-24), then slice to decision date
frames_raw = st._fetch(assets)
frames = {}
for a, df in frames_raw.items():
    if df is None:
        frames[a] = None
        continue
    frames[a] = df[df.index <= pd.Timestamp(CUR)].copy()

scores, used = st._scores(frames, assets, CUR)
print("used factors:", used, "of", len(st.FACTORS))
order = sorted(assets, key=lambda a: (scores[a], a))
print("composite rank (low->high):")
for i, a in enumerate(order):
    print(f"  {i:2d} {a:10s} score={scores[a]:.4f}")

scores = st._de_rank_value_traps(scores, frames, assets, CUR)
regime = st._regime(frames, assets)
w = st._weights(scores, assets, regime)
w = st._composite_top2_cap(w, assets, scores)
w = st._composite_ma_guard(w, frames, assets)
w = st._ma_guard(w, frames, assets, CUR)
for _ in range(6):
    w = st._commod_cap(w, assets)
    w = st._crypto_cap(w, assets)
    w = st._china_cap(w, assets)
    w = st._composite_top2_cap(w, assets, scores)

print("\nregime:", regime)
print("final weights (replay):")
top2 = set(sorted(assets, key=lambda a: (scores[a], a))[-2:])
for a in sorted(w, key=lambda x: -w[x]):
    flag = ""
    if a in top2:
        flag = " [TOP2-COMPOSITE]"
    print(f"  {a:10s} {w[a]*100:6.2f}%{flag}")
print("sum:", round(sum(w.values()), 6))
crypto = w.get("BTC", 0) + w.get("ETH", 0)
comm = w.get("WTI", 0) + w.get("COPPER", 0)
china = w.get("000300.SH", 0) + w.get("000688.SH", 0)
print("BTC+ETH:", round(crypto, 4), "WTI+COPPER:", round(comm, 4), "000300+000688:", round(china, 4))
print("max single:", round(max(w.values()), 4))
