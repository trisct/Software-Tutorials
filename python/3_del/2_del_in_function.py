class Example(object):
    def __del__(self):
        print(f'__del__ is called')

def del_input(x):
    print(f'id(x) = {id(x)}')
    del x

x = Example()
print(f'id(x) = {id(x)}')
print(x, type(x))


del_input(x)
print(f'id(x) = {id(x)}')
print(x, type(x))
