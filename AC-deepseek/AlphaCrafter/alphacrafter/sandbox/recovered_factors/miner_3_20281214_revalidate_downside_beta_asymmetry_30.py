"""Scheduled full revalidation of admitted Cross-Asset Downside Beta Asymmetry 30; visible through 2028-12-13."""
exec(open('scripts/miner_3_20280601_revalidate_downside_beta_asymmetry_30.py').read().replace("E=pd.Timestamp('2028-05-31')", "E=pd.Timestamp('2028-12-13')"))
