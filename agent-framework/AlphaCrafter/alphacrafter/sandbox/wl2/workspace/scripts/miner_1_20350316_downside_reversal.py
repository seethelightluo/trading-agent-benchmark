# same validation, test the opposite orientation: downside-adjusted 30d reversal
exec(open('scripts/miner_1_20350316_downside_efficiency.py').read().replace('f=mom/down;', 'f=-mom/down;').replace('downside_efficiency','downside_reversal'))
