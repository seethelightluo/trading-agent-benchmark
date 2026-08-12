
src = open('strategy.py').read()
i = src.find('def propose')
if i < 0:
    i = src.find('register_hook')
print(src[i:i+3000])
