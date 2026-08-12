"""Dry-run the strategy proposal for 2029-08-31 (no mutation)."""
import json
import sys
sys.path.insert(0, ".")
import strategy as S

cur, tds = S._today_and_calendar()
print("current:", cur, "| should_propose:", S._should_propose(cur, tds))
print("last_proposal:", S._last_proposal_date(tds))
print("FACTORS:", S.FACTORS)

account = S.get_account_dict()
assets = list(account.get("watch_list", []))
print("n assets:", len(assets), "| total_assets:", round(account["total_assets"], 2))

frames = S._fetch(assets)
scores, used = S._scores(frames, assets, cur)
print("factors used:", used)
scores = S._de_rank_value_traps(scores, frames, assets, cur)
regime = S._regime(frames, assets)
print("regime:", regime)
w = S._weights(scores, assets, regime)
w = S._composite_top2_cap(w, assets, scores)
w = S._composite_ma_guard(w, frames, assets)
w = S._ma_guard(w, frames, assets, cur)
w = S._crypto_cap(w, assets)
w = S._commod_cap(w, assets)
print("sum:", round(sum(w.values()), 6))

print("\n--- target weights (sorted) ---")
for a in sorted(w, key=lambda x: -w[x]):
    print(f"{a}: {w[a]*100:.2f}%")

print("\n--- current holdings (market value / weight) ---")
tot = account["total_assets"]
for p in account.get("positions", []):
    print(f"{p['symbol']}: {p['market_value']/tot*100:.2f}%  mv={p['market_value']:.0f}")

print("\n--- momentum ranks & MA state ---")
mom_vals = S._factor_values(frames, "mom_120d_skip5", cur)
mom_rank = S._ranks(mom_vals, assets)
below = S._below_ma(frames, assets)
for a in sorted(assets, key=lambda x: -mom_rank[x]):
    print(f"{a}: mom_rank={mom_rank[a]:.2f} below_ma20={a in below} 120d={mom_vals[a]*100:.1f}%")

print("\n--- forecast returns ---")
f = S._forecasts(scores, assets)
for a in sorted(f, key=lambda x: -f[x]):
    print(f"{a}: {f[a]*100:.2f}%")
