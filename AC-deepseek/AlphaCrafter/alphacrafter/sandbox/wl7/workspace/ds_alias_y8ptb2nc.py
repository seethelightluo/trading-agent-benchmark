import glob,os
syms=['SPX','XAU','WTI','BTC','ETH','N225','HSI','SOX','SX5E','NDX','000300.SH','US10Y','CN10Y','COPPER','000688.SH']
for s in syms:
    try:
        lines=[l.strip().split(',') for l in open(f'../persistent/stock_data/{s}.csv') if l.strip()]
        c=[float(l[1]) for l in lines]
        r=(c[-1]/c[-60]-1)*100
        print(s,'60d_chg %.1f%%'%r,'close',round(c[-1],2))
    except Exception as e:
        print(s,'ERR',e)