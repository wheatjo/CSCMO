import numpy as np
from pymoo.problems.multi.ctp import *
import matplotlib.cm as cm


class DisplayCTP1(CTP1):

    def __init__(self, n_var=2, n_ieq_constr=2, **kwargs):
        super(DisplayCTP1, self).__init__(n_var, n_ieq_constr, **kwargs)
        self.problem_origin = CTP1(n_var=2, n_ieq_constr=2)

    def get_pf_region(self):
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1.25, 400))
        objs = np.dstack((x, y))
        objs.resize((400*400, 2))
        f1, f2 = objs[:, 0], objs[:, 1]

        a, b, e, theta = self.a, self.b, 1, -0.2*np.pi
        g = []
        for j in range(self.n_ieq_constr):
            _g = - (f2 - (a[j] * np.exp(-b[j] * f1)))
            g.append(_g)
        G = np.column_stack(g)
        cv = np.sum(np.maximum(0, G), axis=1)
        feasible_index = (cv <= 0)
        # upf_region = ((f2 - e) * np.cos(theta) - f1 * np.sin(theta) >= 0)
        # upf_region = (f2 >= 1 - np.sqrt(f1))
        upf_region = f2 >= np.exp(-1 * f1)
        z = np.ones_like(x)
        z = np.resize(z, cv.shape)
        z[feasible_index & upf_region] = 0
        z.resize(x.shape)
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem_origin.pareto_front()


class DisplayCTP2(CTP2):

    def __init__(self, n_var=2, n_ieq_constr=1, option="linear"):
        super(DisplayCTP2, self).__init__(n_var=n_var, n_ieq_constr=n_ieq_constr, option="linear")
        self.problem_origin = CTP2(n_var=n_var, n_ieq_constr=n_ieq_constr, option=option)

    def get_pf_region(self):
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 2, 400))
        objs_f = np.dstack((x, y))
        objs_f = objs_f.reshape(400*400, self.n_obj)
        theta = -0.2 * np.pi
        a, b, c, d, e = 0.2, 10, 1, 6, 1
        f1, f2 = objs_f[:, 0], objs_f[:, 1]
        G = self.calc_constraint(theta, a, b, c, d, e, f1, f2)
        feasible_index = np.maximum(0, G) <= 0
        upf_region = ((f2 - e) * np.cos(theta) - f1 * np.sin(theta) >= 0)
        z = np.ones_like(x)
        z = np.resize(z, f1.shape)
        z[feasible_index & upf_region] = 0
        z.resize(x.shape)
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem_origin.pareto_front()


class DisplayCTP3(CTP3):

    def __init__(self, n_var=2, n_ieq_constr=1, option="linear"):
        super(DisplayCTP3, self).__init__(n_var=n_var, n_ieq_constr=n_ieq_constr, option="linear")
        self.problem_origin = CTP3(n_var=n_var, n_ieq_constr=n_ieq_constr, option=option)

    def get_pf_region(self):
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 2, 400))
        objs_f = np.dstack((x, y))
        objs_f = objs_f.reshape(400*400, self.n_obj)
        theta = -0.2 * np.pi
        a, b, c, d, e = 0.1, 10, 1, 0.5, 1
        f1, f2 = objs_f[:, 0], objs_f[:, 1]
        G = self.calc_constraint(theta, a, b, c, d, e, f1, f2)
        feasible_index = np.maximum(0, G) <= 0
        upf_region = ((f2 - e) * np.cos(theta) - f1 * np.sin(theta) >= 0)
        z = np.ones_like(x)
        z = np.resize(z, f1.shape)
        z[feasible_index & upf_region] = 0
        z.resize(x.shape)
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem_origin.pareto_front()


class DisplayCTP4(CTP4):

    def __init__(self, n_var=2, n_ieq_constr=1, option="linear"):
        super(DisplayCTP4, self).__init__(n_var=n_var, n_ieq_constr=n_ieq_constr, option="linear")
        self.problem_origin = CTP4(n_var=n_var, n_ieq_constr=n_ieq_constr, option=option)

    def get_pf_region(self):
        pixs = 400
        x, y = np.meshgrid(np.linspace(0, 1, pixs), np.linspace(0, 2, pixs))
        objs_f = np.dstack((x, y))
        objs_f = objs_f.reshape(pixs*pixs, self.n_obj)
        theta = -0.2 * np.pi
        a, b, c, d, e = 0.75, 10, 1, 0.5, 1
        f1, f2 = objs_f[:, 0], objs_f[:, 1]
        G = self.calc_constraint(theta, a, b, c, d, e, f1, f2)
        feasible_index = np.maximum(0, G) <= 0
        upf_region = ((f2 - e) * np.cos(theta) - f1 * np.sin(theta) >= 0)
        z = np.ones_like(x)
        z = np.resize(z, f1.shape)
        z[feasible_index & upf_region] = 0
        z.resize(x.shape)
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem_origin.pareto_front()


class DisplayCTP5(CTP5):

    def __init__(self, n_var=2, n_ieq_constr=1, option="linear"):
        super(DisplayCTP5, self).__init__(n_var=n_var, n_ieq_constr=n_ieq_constr, option="linear")
        self.problem_origin = CTP5(n_var=n_var, n_ieq_constr=n_ieq_constr, option=option)

    def get_pf_region(self):
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 2, 400))
        objs_f = np.dstack((x, y))
        objs_f = objs_f.reshape(400*400, self.n_obj)
        theta = -0.2 * np.pi
        a, b, c, d, e = 0.1, 10, 2, 0.5, 1
        f1, f2 = objs_f[:, 0], objs_f[:, 1]
        G = self.calc_constraint(theta, a, b, c, d, e, f1, f2)
        feasible_index = np.maximum(0, G) <= 0
        upf_region = ((f2 - e) * np.cos(theta) - f1 * np.sin(theta) >= 0)
        z = np.ones_like(x)
        z = np.resize(z, f1.shape)
        z[feasible_index & upf_region] = 0
        z.resize(x.shape)
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem_origin.pareto_front()


class DisplayCTP6(CTP6):

    def __init__(self, n_var=2, n_ieq_constr=1, option="linear"):
        super(DisplayCTP6, self).__init__(n_var=n_var, n_ieq_constr=n_ieq_constr, option="linear")
        self.problem_origin = CTP6(n_var=n_var, n_ieq_constr=n_ieq_constr, option=option)

    def get_pf_region(self):
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 20, 400))
        objs_f = np.dstack((x, y))
        objs_f = objs_f.reshape(400*400, self.n_obj)
        theta = 0.1 * np.pi
        a, b, c, d, e = 40, 0.5, 1, 2, -2
        f1, f2 = objs_f[:, 0], objs_f[:, 1]
        G = self.calc_constraint(theta, a, b, c, d, e, f1, f2)
        feasible_index = np.maximum(0, G) <= 0
        upf_region = ((f2 - e) * np.cos(theta) - f1 * np.sin(theta) >= 0)
        z = np.ones_like(x)
        z = np.resize(z, f1.shape)
        z[feasible_index & upf_region] = 0
        z.resize(x.shape)
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem_origin.pareto_front()


class DisplayCTP7(CTP7):

    def __init__(self, n_var=2, n_ieq_constr=1, option="linear"):
        super(DisplayCTP7, self).__init__(n_var=n_var, n_ieq_constr=n_ieq_constr, option="linear")
        self.problem_origin = CTP7(n_var=n_var, n_ieq_constr=n_ieq_constr, option=option)

    def get_pf_region(self):
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 2, 400))
        objs_f = np.dstack((x, y))
        objs_f = objs_f.reshape(400*400, self.n_obj)
        theta = -0.05 * np.pi
        a, b, c, d, e = 40, 5, 1, 6, 0
        f1, f2 = objs_f[:, 0], objs_f[:, 1]
        G = self.calc_constraint(theta, a, b, c, d, e, f1, f2)
        feasible_index = np.maximum(0, G) <= 0
        # upf_region = (((f2 - e) * np.cos(theta) + f1 * np.sin(theta) >= 1) & ((f2 - e) * np.cos(theta) + f1 * np.sin(theta) <= 2))
        # upf_region = ((f2 - 0.3) * np.cos(theta) - f1 * np.sin(theta) >= 0)
        # upf_region = (f2 >= (1 + f1) * (1 - np.sqrt(2.05) * np.sqrt(f1/(1 + f1))))
        upf_region = (f2 >= 1 - np.sqrt(f1))
        z = np.ones_like(x)
        z = np.resize(z, f1.shape)
        z[feasible_index & upf_region] = 0
        z.resize(x.shape)
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem_origin.pareto_front()


class DisplayCTP8(CTP8):

    def __init__(self):
        super(DisplayCTP8, self).__init__(option="linear")
        self.problem_origin = CTP8()

    def get_pf_region(self):
        x, y = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 5, 400))
        objs_f = np.dstack((x, y))
        objs_f = objs_f.reshape(400*400, self.n_obj)
        f1, f2 = objs_f[:, 0], objs_f[:, 1]
        theta = 0.1 * np.pi
        a, b, c, d, e = 40, 0.5, 1, 2, -2
        g1 = self.calc_constraint(theta, a, b, c, d, e, f1, f2)

        theta = -0.05 * np.pi
        a, b, c, d, e = 40, 2, 1, 6, 0
        g2 = self.calc_constraint(theta, a, b, c, d, e, f1, f2)
        G = np.column_stack([g1, g2])
        feasible_index = np.sum(np.maximum(0, G), axis=1) <= 0
        upf_region = (f2 >= 1 - np.sqrt(f1))
        # upf_region = ((f2 - e) * np.cos(theta) - f1 * np.sin(theta) >= 0)
        z = np.ones_like(x)
        z = np.resize(z, f1.shape)
        z[feasible_index & upf_region] = 0
        z.resize(x.shape)
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem_origin.pareto_front()


if __name__ == '__main__':

    import matplotlib.pyplot as plt



    A = DisplayCTP4()
    X, Y, Z = A.get_pf_region()
    # Z = interpolate.interp2d(X, Y, Z, kind='cubic')
    fig, ax = plt.subplots()
    fig.set_size_inches(20.5, 20.5, forward=True)
    fig.set_dpi(300)
    im = ax.imshow(Z, cmap=cm.gray,
                   origin='lower', extent=[X[0][0], X[0][-1], Y[0][0], Y[-1][0]],
                   vmax=abs(Z / 9).max(), vmin=-abs(Z).max(), aspect='auto', interpolation='antialiased')
    pf = A.pareto_front()
    x = np.linspace(0, 1, 400)
    # y = (1 + x) * np.exp(-1 * (x/1+x))
    # y = (1 + x) * (1 - np.sqrt(2.05) * np.sqrt(x/(1 + x)))

    ax.scatter(pf[:, 0], pf[:, 1])

    plt.savefig('ctp4.png')

    # plt.show()


