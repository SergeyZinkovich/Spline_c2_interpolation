import numpy as np
from sympy import *


class Spline:

    def __init__(self, x, f):  # точки и значения функции в них
        self.x = x
        self.f = f

    def set_first_boundary_condition(self, df_a, df_b):  # первая производная на границах
        self.df_a = df_a
        self.df_b = df_b
        self.find_b_with_first_boundary_condition()

    def set_second_boundary_condition(self, ddf_a, ddf_b):  # вторая производная на границах
        self.ddf_a = ddf_a
        self.ddf_b = ddf_b
        self.find_b_with_second_boundary_condition()

    def set_third_boundary_condition(self):  # функция переодическая
        self.find_b_with_third_boundary_condition()

    def h(self, i):
        return self.x[i+1] - self.x[i]

    def lambd(self, i):
        return self.h(i)/(self.h(i - 1) + self.h(i))

    def nu(self, i):
        return 1 - self.lambd(i)

    def find_b_with_first_boundary_condition(self):
        a = np.zeros((len(self.x) - 1, len(self.x) - 1))
        a[0][0] = 1 + self.lambd(1)
        a[0][1] = self.nu(1)

        for i in range(1, len(self.x) - 2):
            a[i][i - 1] = self.lambd(i)
            a[i][i] = 1 + self.nu(i) + self.lambd(i + 1)
            a[i][i + 1] = self.nu(i + 1)

        a[-1][-2] = self.lambd(len(self.x) - 2)
        a[-1][-1] = 1 + self.nu(len(self.x) - 2)

        d = np.array([3 * (self.f[i] - self.f[i - 1]) / (self.x[i] - self.x[i - 1]) for i in range(1, len(self.x) - 1)])
        d[0] = 3 * (self.f[1] - self.f[0]) / (self.x[1] - self.x[0]) - self.df_a
        d = np.append(d, [3 * (self.f[len(self.x) - 1] - self.f[len(self.x) - 2]) / (self.x[len(self.x) - 1] - self.x[len(self.x) - 2]) - self.df_b])

        self.b = np.linalg.solve(a, d)
        self.b = np.insert(self.b, 0, self.df_a)
        self.b = np.append(self.b, self.df_b)

    def find_b_with_second_boundary_condition(self):
        a = np.zeros((len(self.x) - 1, len(self.x) - 1))
        a[0][0] = 2 + self.lambd(1)
        a[0][1] = self.nu(1)

        for i in range(1, len(self.x) - 2):
            a[i][i - 1] = self.lambd(i)
            a[i][i] = 1 + self.nu(i) + self.lambd(i + 1)
            a[i][i + 1] = self.nu(i + 1)

        a[-1][-2] = self.lambd(len(self.x) - 2)
        a[-1][-1] = 2 + self.nu(len(self.x) - 2)

        d = np.array([3 * (self.f[i] - self.f[i - 1]) / (self.x[i] - self.x[i - 1]) for i in range(1, len(self.x) - 1)])
        d[0] = 3 * (self.f[1] - self.f[0]) / (self.x[1] - self.x[0]) + self.ddf_a * self.h(0) / 2
        d = np.append(d, [3 * (self.f[len(self.x) - 1] - self.f[len(self.x) - 2]) / (self.x[len(self.x) - 1] - self.x[len(self.x) - 2]) - self.ddf_b * self.h(len(self.x) - 2) / 2])

        self.b = np.linalg.solve(a, d)
        self.b = np.insert(self.b, 0, -self.h(0) * self.ddf_a / 2 + self.b[0])
        self.b = np.append(self.b, self.h(len(self.x) - 2) * self.ddf_b / 2 + self.b[-1])

    def find_b_with_third_boundary_condition(self):
        a = np.zeros((len(self.x) - 1, len(self.x) - 1))
        #a[0][0] = 1 + self.nu(0) + self.lambd(1)
        a[0][0] = 1 + self.lambd(1)
        a[0][1] = self.nu(1)
        #a[0][-1] = self.lambd(len(self.x) - 1)
        a[0][-1] = 0

        for i in range(1, len(self.x) - 2):
            a[i][i - 1] = self.lambd(i)
            a[i][i] = 1 + self.nu(i) + self.lambd(i + 1)
            a[i][i + 1] = self.nu(i + 1)

        #a[-1][0] = self.nu(0)
        a[-1][0] = 0
        a[-1][-2] = self.lambd(len(self.x) - 2)
        #a[-1][-1] = 1 + self.nu(len(self.x) - 2) + self.lambd(len(self.x) - 1)
        a[-1][-1] = 1 + self.nu(len(self.x) - 2)

        d = np.array([3 * (self.f[i] - self.f[i - 1]) / (self.x[i] - self.x[i - 1]) for i in range(1, len(self.x))])

        self.b = np.linalg.solve(a, d)

    def interpolate_at_point(self, xx):
        for h in range(len(self.x) - 1):
            if((self.x[h] <= xx) & (self.x[h + 1] >= xx)):
                i = h
                break
        t = (xx - (self.x[i])) / self.h(i)
        return (1 - t**3) * self.f[i] + t**3 * self.f[i + 1] + t * (1 - t) * self.lambd(i) * self.h(i) * self.b[i] + t * (1 - t) * (self.nu(i) + t) * self.h(i) * self.b[i + 1]

    def spline_as_string(self, i):
        x = symbols('x')
        return str(simplify((1 - (x - (self.x[i])) / self.h(i)**3) * self.f[i] + (x - (self.x[i])) / self.h(i)**3 * self.f[i + 1] +
                 (x - (self.x[i])) / self.h(i) * (1 - (x - (self.x[i])) / self.h(i)) * self.lambd(i) * self.h(i) * self.b[i] +
                 (x - (self.x[i])) / self.h(i) * (1 - (x - (self.x[i])) / self.h(i)) * (self.nu(i) + (x - (self.x[i])) / self.h(i))
                 * self.h(i) * self.b[i + 1]))

    def all_splines_as_string(self):
        result = ""
        for i in range(len(self.x) - 2):
            result = result + self.spline_as_string(i) + "\n"
        return result




