"""miner_1 candidate: residualized crypto-basket correlation decoupling (60/20)."""
import pathlib
p = pathlib.Path('scripts/miner_1_20280127_residualized_inflation_basket_correlation_decoupling_60_20.py')
s = p.read_text()
s = s.replace("END=pd.Timestamp('2028-01-26')", "END=pd.Timestamp('2028-02-09')")
s = s.replace('residualized inflation-basket correlation decoupling', 'residualized crypto-basket correlation decoupling')
s = s.replace('residualized_inflation_basket_correlation_decoupling', 'residualized_crypto_basket_correlation_decoupling')
s = s.replace("inflation=r[['COPPER','WTI']].mean(axis=1)", "inflation=r[['BTC','ETH']].mean(axis=1)")
p_out = pathlib.Path('scripts/miner_1_20280210_residualized_crypto_basket_correlation_decoupling_60_20.py')
p_out.write_text(s)
print(p_out)
