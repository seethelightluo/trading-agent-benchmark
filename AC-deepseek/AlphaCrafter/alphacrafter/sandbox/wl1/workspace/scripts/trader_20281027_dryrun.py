"""Trader dry-run 2028-10-27: compute proposed target without submitting orders."""
import json
import strategy as S

cur, tds = S._today_and_calendar()
print("current_date:", cur, "| should_propose:", S._should_propose(cur, tds))

acc = __import__('alphacrafter.sim.utils', fromlist=['get_account_dict']).get_account_dict()
assets = list(acc.get('watch_list', []))
print("n_assets:", len(assets))

frames = S._fetch(assets)
scores, used = S._scores(frames, assets, cur)
print("factors used:", used)
for a in sorted(scores, key=lambda x: -scores[x]):
    print(f"  {a:8s} score={scores[a]:.4f}")

scores = S._de_rank_value_traps(scores, frames, assets, cur)
regime = S._regime(frames, assets)
print("regime:", regime)
w = S._weights(scores, assets, regime)
w = S._composite_ma_guard(w, frames, assets)
w = S._ma_guard(w, frames, assets, cur)
w = S._crypto_cap(w, assets)
w = S._commod_cap(w, assets)

print("--- proposed target ---")
tot = 0.0
for a in sorted(w, key=lambda x: -w[x]):
    print(f"  {a:8s} {w[a]*100:6.2f}%")
    tot += w[a]
print("sum:", round(tot, 10))

# guard checks
crypto = sum(w[a] for a in assets if a in S.CRYPTO)
comm = sum(w[a] for a in assets if a in S.CYCLICAL_COMMOD)
below = S._below_ma(frames, assets)
mom_vals = S._factor_values(frames, 'mom_120d_skip5', cur)
mom_rank = S._ranks(mom_vals, assets)
top2 = sorted(assets, key=lambda a: -mom_rank[a])[:2]
print("crypto combined:", round(crypto, 4), "| comm combined:", round(comm, 4))
print("mom top2:", [(a, round(mom_rank[a], 3)) for a in top2])
print("top2 weights:", {a: round(w[a], 4) for a in top2})
print("below MA20:", sorted(below))
