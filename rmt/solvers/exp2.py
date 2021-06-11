

class fun1:
    def __init__(self, a):
        print("class init")
        self.a = a

    def __call__(self, x):
        return x + self.a

    def __str__(self) -> str:
        return "Sina changed this function as"

    def id(self):
        return 'a=%a' % self.a

# y = textC()
# z = y(1)
# print(z)


# def f(x):
#     return x + 1


# def df(f, x):
#     return f(x)


# ans = df(f, 10)
# print(ans)

# use function with args
f2 = fun1(2)
print(f2)


# ans = df(f2, 10)
# print(ans)

# id = f2.id()
# print(id)
