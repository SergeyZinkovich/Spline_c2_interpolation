from main import Spline
import matplotlib.pyplot as plt
import numpy as np

def find_max_error(a, b):
    ans = 0
    for i in range(len(a)):
        ans = max(ans, abs(a[i] - b[i]))
    return ans

def test(f, df, a, b):
    x1 = np.linspace(a, b, 10)
    s1 = Spline(x1, f(x1))
    s1.set_first_boundary_condition(df(x1[0]), df(x1[-1]))
    y = []
    test_x = np.linspace(a, b, 1000)
    for i in test_x:
        y.append(s1.interpolate_at_point(i))

    print(find_max_error(f(test_x), y))
    print(s1.all_splines_as_string())

    fig, ax = plt.subplots()
    ax.plot(test_x, f(test_x), color="red", label="sin(x)")
    ax.plot(test_x, y, color="blue", label="interpolated")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()

test(np.sin, np.cos, 0, 2 * np.pi)

def f1(x):
    return 3 * x**2

def df1(x):
    return 6*x

test(f1, df1, -1, 1)

def dabs(x):
    return x/abs(x)

test(np.abs, dabs, -1, 1)
