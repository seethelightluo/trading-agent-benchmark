import os
print("CWD:", os.getcwd())
print("factor_ensemble:", open('factor_ensemble.json').read())
print("===MEM tail===")
print(open('memory.txt').read()[-3000:])