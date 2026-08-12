"""Trader preview: compute today's (2031-01-17) would-be proposal without executing.

Validates ensemble load, factor availability, regime, scores, and final
weight vector invariants (non-negative, sum to 1, 15 assets, no cash).
Read-only: does NOT call step/backtest/rebalance_to_weights.
"""
import json
import sys

sys.path.insert(0, ".")
import strategy as S

# Force ensemble reload to match factor_ensemble.json
S.FACTORS = S._load_ensemble()
print("Ensemble:")
for fid, w, d in S.FACTORS:
    print(f"  {fid:28s} w={w:.2f} dir={d:+d}")

cur, tds = S._today_and_calendar()
print("\ncurrent_date:", cur, "| is proposal day:", S._should_propose(cur, tds))

acc = S.get_account_dict()
assets = list(acc.get("watch_list", []))
print("assets:", len(assets), assets)

frames = S._fetch(assets)
missing = [a for a, df in frames.items() if df is None or len(df) < S.MIN_ROWS]
print("missing/short frames:", missing)

scores, used = S._scores(frames, assets, cur)
print("factors used:", used)
print("\ncomposite scores (pre guards):")
for a in sorted(assets, key=lambda x: -scores[x]):
    print(f"  {a:10s} {scores[a]:+.4f}")

scores = S._de_rank_value_traps(scores, frames, assets, cur)
regime = S._regime(frames, assets)
print("\nregime:", regime)

w = S._weights(scores, assets, regime)
w = S._composite_top2_cap(w, assets, scores)
w = S._composite_ma_guard(w, frames, assets)
w = S._ma_guard(w, frames, assets, cur)
for _ in range(6):
    w = S._commod_cap(w, assets)
    w = S._crypto_cap(w, assets)

print("\nfinal target weights:")
total = 0.0
for a in sorted(assets, key=lambda x: -w[x]):
    print(f"  {a:10s} {w[a]:.4f}")
    total += w[a]
print("sum:", round(total, 8), "| min:", min(w.values()), "| max:", max(w.values()))
print("BTC+ETH:", round(w.get("BTC", 0) + w.get("ETH", 0), 4), "<= 0.12:",
      w.get("BTC", 0) + w.get("ETH", 0) <= 0.12 + 1e-9)
print("WTI+COPPER:", round(w.get("WTI", 0) + w.get("COPPER", 0), 4), "<= 0.14:",
      w.get("WTI", 0) + w.get("COPPER", 0) <= 0.14 + 1e-9)
print("defensive (XAU+US10Y+CN10Y):", round(sum(w[a] for a in S.DEFENSIVE), 4))

below = S._below_ma(frames, assets)
print("below MA20:", sorted(below))
