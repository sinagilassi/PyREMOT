

class fun1:

    # test
    testVal = 1

    def __init__(self, a):
        print("class init")
        self.a = a

    def __call__(self, x):
        return x + self.a

    def __str__(self) -> str:
        return "Sina changed this function as"

    def id(self):
        return 'a=%a' % self.a

    @classmethod
    def updateValue1(cls, b):
        cls.testVal = b

    def updateValue2(self, b):
        self.testVal = b

    @property
    def showValue(cls):
        print("testVal: ", cls.testVal)
        return cls.testVal

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
# f2 = fun1(2)
# print(f2)

# val = f2.updateValue2(100)

# val2 = f2.showValue
# print(fun1.a)


# ans = df(f2, 10)
# print(ans)

# id = f2.id()
# print(id)

a = [1, 2, 3]
b = a.copy()
c = [3, 4, 5]
print(a, b)
a.clear()
print(a, b)
a.extend(c)
print(a, b)
