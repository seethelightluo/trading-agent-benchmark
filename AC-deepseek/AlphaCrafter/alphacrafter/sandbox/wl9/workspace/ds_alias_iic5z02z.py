import os
print("CWD:", os.getcwd())
print(os.listdir('.'))
print("---factors dir---")
print([f for f in os.listdir('factors/')])
print("---ensemble any?---")
for root,dirs,files in os.walk('.'):
    for f in files:
        if 'ensemble' in f or f=='memory.txt':
            print(os.path.join(root,f))