# miner_3 recent explorations to avoid duplication
for f in ['scripts/miner_3_20280727_clv_location.py','scripts/miner_3_20280713_candle_body_ratio.py','scripts/miner_3_20280420_drawdown_gap.py','scripts/miner_3_20280323_explore_short_reversal.py']:
    lines = open(f).read().splitlines()
    doc = ' '.join(l.strip() for l in lines[:3] if l.strip() and not l.strip().startswith(('import','from','"""','#')))
    print(f"{f}:\n  {doc[:220]}")