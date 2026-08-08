# Execute same factor definition and report the selected 20-session horizon by regime.
exec(open('scripts/miner_1_20330512_trend_orthogonal_mild_pullback_capture_60.py').read())
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME20',n,stats(20,p))
