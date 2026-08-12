lines = open('memory.txt').read().splitlines()
for l in lines[-6:]:
    print(l[:600])
    print('===')
