"""Trader dry-run: compute the 2028-10-13 proposal target weights without
calling rebalance_to_weights (safe - no account mutation)."""
import json
import strategy as st

cur, tds = st._today_and_calendar()
print("current_date:", cur)
print("is_grid_rebalance_day:", st._is_rebalance_day(cur, tds))
print("last_proposal:", st._last_proposal_date(tds))
print("should_propose:", st._should_propose(cur, tds))

acc = json.load(open('../persistent/account.json'))
assets = acc['watch_list']
frames = st._fetch(assets)
scores, used = st._scores(frames, assets, cur)
print("factors_used:", used, "of", len(st.FACTORS))
for fid, w, d in st.FACTORS:
    vals = st._factor_values(frames, fid, cur)
    nv = sum(1 for v in vals.values() if v is not None)
    print(f"  {fid} w={w} dir={d} valid={nv}/15")

scores = st._de_rank_value_traps(scores, frames, assets, cur)
regime = st._regime(frames, assets)
print("regime:", regime)
w = st._weights(scores, assets, regime)
w = st._composite_ma_guard(w, frames, assets)
w = st._ma_guard(w, frames, assets, cur)
w = st._crypto_cap(w, assets)
w = st._commod_cap(w, assets)

tot = sum(w.values())
print("sum:", round(tot, 10), "min:", min(w.values()), "max:", max(w.values()))
for a in sorted(w, key=lambda x: -w[x]):
    print(f"  {a:10s} {w[a]*100:6.2f}%")
print("crypto_sum:", round((w['BTC']+w['ETH'])*100, 2), "%")
print("comm_sum:", round((w['WTI']+w['COPPER'])*100, 2), "%")

f = st._forecasts(scores, assets)
print("forecast range:", min(f.values()), max(f.values()))
